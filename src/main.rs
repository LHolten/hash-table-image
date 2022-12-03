#![feature(array_zip)]
#![feature(iter_array_chunks)]
#![feature(array_chunks)]
#![feature(generic_const_exprs)]
#![feature(new_uninit)]

use std::{fs::File, io::BufReader};

use dfdx::prelude::*;
use jpeg_decoder as jpeg;
use rand::{rngs::StdRng, SeedableRng};
use show_image::{create_window, ImageInfo, ImageView};
// use vecmath::{vec2_add, vec2_mul, vec2_scale, vec2_sub};

use crate::hashtable::HashTable;

mod box_slice;
mod hashtable;

// pub fn render_layer<const L: usize, const T: usize>(
//     table: HashTable<L, T>,
//     layer: usize,
//     feature: usize,
// ) -> Box<[[u8; 2]; 256 * 256]> {
//     let coords = (0..256).flat_map(|y| (0..256).map(move |x| [x, y]));

//     let values: Box<[[f32; 2]; 256 * 256]> = coords
//         .array_chunks()
//         .flat_map(|coords| {
//             let output: Tensor2D<256, 2, NoneTape> = table.get(&coords, layer);
//             *output.data()
//         })
//         .collect::<Box<[[f32; 2]]>>()
//         .try_into()
//         .unwrap();
//     let total = values.iter().copied().fold([0., 0.], vec2_add);
//     let avg = vec2_scale(total, 1. / (256. * 256.));
//     values.iter_mut().for_each(|v| *v = vec2_sub(*v, avg));
//     let total_var = values
//         .iter()
//         .copied()
//         .map(|x| vec2_mul(x, x))
//         .fold([0., 0.], vec2_add);
//     let var = vec2_scale(total_var, 1. / (256. * 256.));
//     // values.into_iter()

//     // let total_squared = values.iter().sum();

//     // values.iter_mut().for_each(|v| *v /= avg)
// }

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut table = HashTable::<7, { 1 << 10 }>::default();

    let mut rng = StdRng::seed_from_u64(0);
    table.reset_params(&mut rng);

    println!("{:?}", table.grid_sizes());

    let file = File::open("SIPI_Jelly_Beans_4.1.07.tiff.jpg").expect("failed to open file");
    let mut decoder = jpeg::Decoder::new(BufReader::new(file));
    let pixels = decoder.decode().expect("failed to decode image");

    let image = ImageView::new(ImageInfo::rgb8(256, 256), &pixels);
    let window = create_window("image", Default::default())?;
    window.set_image("reference", image)?;

    let pixels: Box<[[f32; 3]; 256 * 256]> = pixels
        .chunks_exact(3)
        .map(|c| [0, 1, 2].map(|i| c[i] as f32 / 255.))
        .collect::<Box<[[f32; 3]]>>()
        .try_into()
        .expect("image has the wrong size");

    let coords: Box<[[usize; 2]; 256 * 256]> = (0..256)
        .flat_map(|y| (0..256).map(move |x| [x, y]))
        .collect::<Box<[[usize; 2]]>>()
        .try_into()
        .unwrap();

    let mut opt = Adam::new(AdamConfig {
        lr: 10f32.powi(-2),
        betas: [0.9, 0.99],
        eps: 10f32.powi(-15),
        weight_decay: None,
    });

    loop {
        let mut img = Vec::with_capacity(256 * 256);
        let loss = coords
            .array_chunks::<{ 256 * 64 }>()
            .zip(pixels.array_chunks())
            .map(|(coords, pixels)| {
                let pixels = TensorCreator::new(*pixels);
                let output = table.forward(coords);
                img.extend(output.data().iter().flatten().map(|v| (v * 255.) as u8));
                mse_loss(output, pixels)
            })
            .fold(Tensor0D::new(0.).with_diff_tape(), |a, b| b + a);
        let loss = div_scalar(loss, 4.);
        println!("loss = {}", loss.data() * 256.);
        let grad = backward(loss);

        opt.update(&mut table, grad).expect("some unused gradients");

        let image = ImageView::new(ImageInfo::rgb8(256, 256), &img);
        window.set_image("output", image)?;
    }
}
