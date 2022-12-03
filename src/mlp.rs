use dfdx::{
    gradients::{CanUpdateWithGradients, Tape},
    prelude::*,
};
use rand::distributions::Uniform;

pub type Mlp<const L: usize> = (ReLU, Linear<32, 32>, ReLU, Linear<32, 3>, Sigmoid);

#[derive(Clone)]
pub struct ConcatLinear<const L: usize, const O: usize> {
    weights: [Tensor2D<O, 2>; L],
    bias: Tensor1D<O>,
}

impl<const B: usize, const L: usize, const O: usize, H: Tape> Module<[Tensor2D<B, 2, H>; L]>
    for ConcatLinear<L, O>
{
    type Output = Tensor2D<B, O, H>;

    fn forward(&self, input: [Tensor2D<B, 2, H>; L]) -> Self::Output {
        let mut output = self.bias.clone().with_diff_tape().broadcast();
        for (x, w) in input.zip(self.weights.clone()) {
            output = output + matmul_transpose(x, w)
        }
        output
    }
}

impl<const L: usize, const O: usize> ResetParams for ConcatLinear<L, O> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        let bound: f32 = 1.0 / ((L * 2) as f32).sqrt();
        let dist = Uniform::new(-bound, bound);
        self.weights
            .iter_mut()
            .for_each(|w| w.randomize(rng, &dist));
        self.bias.randomize(rng, &dist);
    }
}

impl<const L: usize, const O: usize> CanUpdateWithGradients for ConcatLinear<L, O> {
    fn update<G: dfdx::gradients::GradientProvider>(
        &mut self,
        grads: &mut G,
        unused: &mut dfdx::gradients::UnusedTensors,
    ) {
        self.weights
            .iter_mut()
            .for_each(|w| w.update(grads, unused));
        self.bias.update(grads, unused);
    }
}

impl<const L: usize, const O: usize> Default for ConcatLinear<L, O> {
    fn default() -> Self {
        let weights = std::array::from_fn(|_| Default::default());
        Self {
            weights,
            bias: Default::default(),
        }
    }
}
