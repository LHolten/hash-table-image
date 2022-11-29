#![feature(array_zip)]

use dfdx::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

use crate::hashtable::HashTable;

mod hashtable;
mod mlp;

fn main() {
    let mut table = HashTable::<16, { 1 << 14 }>::default();

    let mut rng = StdRng::seed_from_u64(0);
    table.reset_params(&mut rng);

    // let img = Vec<{256 * 256}>::

    // let grad = ;
    for (coord, col) in img {
        let res = table.forward(coord);
        let loss = mse_loss(res, col);
        backward(loss)
    }

    println!("Hello, world!");
}
