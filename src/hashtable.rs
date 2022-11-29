use dfdx::{
    gradients::{CanUpdateWithGradients, Tape},
    prelude::*,
};
use rand::distributions::Uniform;

use crate::mlp::{ConcatLinear, Mlp};

pub struct HashTable<const L: usize, const T: usize> {
    max_res: usize,
    min_res: usize,
    layers: [Tensor2D<T, 2>; L],
    concat_linear: ConcatLinear<L, 64>,
    mlp: Mlp<L>,
}

impl<const L: usize, const T: usize> HashTable<L, T> {
    fn coord_to_index(&self, res: usize, coord: [usize; 2]) -> usize {
        const PRIME: usize = 2_654_435_761;
        if res * res > T {
            (coord[0] ^ coord[1].wrapping_mul(PRIME)) % T
        } else {
            coord[0] + coord[1] * res
        }
    }

    const CORNERS: [[usize; 2]; 4] = [[0, 0], [0, 1], [1, 1], [1, 0]];

    fn get<const B: usize, H: Tape>(
        &self,
        coords: &[[usize; 2]; B],
        layer: usize,
    ) -> Tensor2D<B, 2, H> {
        let res = self.grid_sizes()[layer];
        let factor = res as f32 / self.max_res as f32;
        let scaled_coords = coords
            .iter()
            .map(|&[x, y]| [x as f32 * factor, y as f32 * factor])
            .collect::<Vec<_>>();

        let indices = scaled_coords
            .iter()
            .map(|&[x, y]| {
                let [x, y] = [x.floor() as usize, y.floor() as usize];
                Self::CORNERS.map(|up| self.coord_to_index(res, [x + up[0], y + up[1]]))
            })
            .collect::<Vec<_>>();
        let indices: Box<[[usize; 4]; B]> = indices.into_boxed_slice().try_into().unwrap();

        let weights = scaled_coords
            .iter()
            .map(|&[x, y]| {
                let [x, y] = [x.fract(), y.fract()];
                let x_w = [1. - x, x];
                let y_w = [1. - y, y];
                Self::CORNERS.map(|up| x_w[up[0]] * y_w[up[1]])
            })
            .collect::<Vec<_>>();
        let weights: Box<[[f32; 4]; B]> = weights.into_boxed_slice().try_into().unwrap();

        let weights: Tensor2D<B, 4> = TensorCreator::new_boxed(weights);
        let weights: Tensor3D<B, 4, 2> = weights.broadcast();

        let layer = self.layers[layer].with_diff_tape();
        let selected: Tensor3D<B, 4, 2, H> = layer.select(&*indices);
        mul(selected, weights).sum()
    }

    fn grid_sizes(&self) -> [usize; L] {
        let d = (self.max_res as f32).ln() - (self.min_res as f32).ln();
        let b = (d / (L as f32 - 1.)).exp();
        std::array::from_fn(|i| (self.min_res as f32 * b.powi(i as i32)) as usize)
    }
}

impl<const L: usize, const T: usize> HashTable<L, T> {
    pub fn forward<const B: usize, H: Tape>(&self, input: &[[usize; 2]; B]) -> Tensor2D<B, 3, H> {
        let features = std::array::from_fn(|i| self.get(input, i));
        let concat = self.concat_linear.forward(features);
        self.mlp.forward(concat)
    }
}

impl<const L: usize, const T: usize> ResetParams for HashTable<L, T> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.concat_linear.reset_params(rng);
        self.mlp.reset_params(rng);

        let dist = Uniform::new(-0.0001, 0.0001);
        self.layers.iter_mut().for_each(|w| w.randomize(rng, &dist));
    }
}

impl<const L: usize, const T: usize> CanUpdateWithGradients for HashTable<L, T> {
    fn update<G: dfdx::gradients::GradientProvider>(
        &mut self,
        grads: &mut G,
        unused: &mut dfdx::gradients::UnusedTensors,
    ) {
        self.concat_linear.update(grads, unused);
        self.mlp.update(grads, unused);
        self.layers.iter_mut().for_each(|w| w.update(grads, unused));
    }
}

impl<const L: usize, const T: usize> Default for HashTable<L, T> {
    fn default() -> Self {
        let layers = std::array::from_fn(|_| Default::default());
        Self {
            max_res: 256,
            min_res: 4,
            layers,
            concat_linear: Default::default(),
            mlp: Default::default(),
        }
    }
}
