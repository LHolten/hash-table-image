use std::iter::zip;

use dfdx::{
    gradients::{CanUpdateWithGradients, Tape},
    prelude::*,
};
use rand::distributions::Uniform;

use crate::box_slice::map_boxed2;

pub struct HashTable<const L: usize, const T: usize>
where
    [(); L * 2]:,
{
    max_res: usize,
    min_res: usize,
    layers: Tensor3D<L, T, 2>,
    mlp: (
        Linear<{ L * 2 }, 32>,
        ReLU,
        Linear<32, 32>,
        ReLU,
        Linear<32, 3>,
        Sigmoid,
    ),
}

impl<const L: usize, const T: usize> HashTable<L, T>
where
    [(); L * 2]:,
{
    fn coord_to_index(&self, res: usize, coord: [usize; 2]) -> usize {
        const PRIME: usize = 2_654_435_761;
        debug_assert!(coord[0] <= res, "{} out of bound {res}", coord[0]);
        debug_assert!(coord[1] <= res, "{} out of bound {res}", coord[1]);
        if res * (res + 2) >= T {
            (coord[0] ^ coord[1].wrapping_mul(PRIME)) % T
        } else {
            coord[0] + coord[1] * (res + 1)
        }
    }

    const CORNERS: [[usize; 2]; 4] = [[0, 0], [0, 1], [1, 1], [1, 0]];

    pub fn get<const B: usize, H: Tape>(&self, coords: &[[usize; 2]; B]) -> Tensor3D<L, B, 2, H> {
        let all_res = self.grid_sizes();

        let scaled_coords: Box<[[[f32; 2]; B]; L]> = map_boxed2(
            all_res,
            |_| coords,
            |res, &[x, y]| {
                let factor = res as f32 / self.max_res as f32;
                [x as f32 * factor, y as f32 * factor]
            },
        );

        let weighted: [Tensor3D<L, B, 2, H>; 4] = Self::CORNERS.map(|up| {
            let indices = map_boxed2(
                zip(all_res, &*scaled_coords),
                |(_, s)| s,
                |(res, _), &[x, y]| {
                    let [x, y] = [x.floor() as usize, y.floor() as usize];
                    self.coord_to_index(res, [x + up[0], y + up[1]])
                },
            );

            let weights = map_boxed2(
                &*scaled_coords,
                |s| s,
                |_, &[x, y]| {
                    let [x, y] = [x.fract(), y.fract()];
                    let x_w = [1. - x, x];
                    let y_w = [1. - y, y];
                    x_w[up[0]] * y_w[up[1]]
                },
            );

            let weights: Tensor2D<L, B> = TensorCreator::new_boxed(weights);
            let weights: Tensor3D<L, B, 2> = weights.broadcast();

            let selected = self.layers.with_diff_tape().select(&*indices);
            mul(selected, weights)
        });

        let [a, b, c, d] = weighted;
        a + b + c + d
    }

    pub fn grid_sizes(&self) -> [usize; L] {
        let d = (self.max_res as f32).ln() - (self.min_res as f32).ln();
        let b = (d / (L as f32 - 1.)).exp();
        std::array::from_fn(|i| (self.min_res as f32 * b.powi(i as i32)) as usize)
    }
}

impl<const L: usize, const T: usize> HashTable<L, T>
where
    [(); L * 2]:,
{
    pub fn forward<const B: usize, H: Tape>(&self, input: &[[usize; 2]; B]) -> Tensor2D<B, 3, H>
    where
        Tensor3D<B, L, 2, H>: Reshape<Tensor2D<B, { L * 2 }, H>>,
    {
        let features = self.get(input);
        let concat: Tensor3D<B, L, 2, H> = features.permute();
        let concat: Tensor2D<B, { L * 2 }, H> = concat.reshape();
        self.mlp.forward(concat)
    }
}

impl<const L: usize, const T: usize> ResetParams for HashTable<L, T>
where
    [(); L * 2]:,
{
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.mlp.reset_params(rng);

        let dist = Uniform::new(-0.0001, 0.0001);
        self.layers.randomize(rng, &dist);
    }
}

impl<const L: usize, const T: usize> CanUpdateWithGradients for HashTable<L, T>
where
    [(); L * 2]:,
{
    fn update<G: dfdx::gradients::GradientProvider>(
        &mut self,
        grads: &mut G,
        unused: &mut dfdx::gradients::UnusedTensors,
    ) {
        self.mlp.update(grads, unused);
        self.layers.update(grads, unused);
    }
}

impl<const L: usize, const T: usize> Default for HashTable<L, T>
where
    [(); L * 2]:,
{
    fn default() -> Self {
        Self {
            max_res: 256,
            min_res: 4,
            layers: Default::default(),
            mlp: Default::default(),
        }
    }
}
