#![feature(iter_array_chunks)]
#![feature(array_chunks)]
#![feature(generic_const_exprs)]
#![feature(new_uninit)]

use std::{fs::File, time::Instant};

use dfdx::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use show_image::{
    create_window,
    event::{ElementState, VirtualKeyCode, WindowEvent},
    ImageInfo, ImageView,
};
use tiff::decoder::{Decoder, DecodingResult};

use crate::hashtable::HashTable;

mod box_slice;
mod hashtable;

pub fn render_layer<const L: usize, const T: usize>(
    table: &HashTable<L, T>,
    res: u32,
) -> ([Vec<u8>; L * 2], Vec<u8>)
where
    [Vec<f32>; L * 2]: Default,
    Tensor3D<256, L, 2>: Reshape<Tensor2D<256, { L * 2 }>>,
{
    let coords = (0..res).flat_map(|y| {
        (0..res).map(move |x| [(x as f32 + 0.5) / res as f32, (y as f32 + 0.5) / res as f32])
    });

    let mut values: [Vec<f32>; L * 2] = Default::default();

    coords.clone().array_chunks().for_each(|coords| {
        let output: Tensor3D<L, 256, 2> = table.get(&coords);
        let output: Tensor3D<L, 2, 256> = output.permute();

        for l in 0..L {
            for f in 0..2 {
                values[l * 2 + f].extend_from_slice(&output.data()[l][f])
            }
        }
    });

    for layer in &mut values {
        let total: f32 = layer.iter().sum();
        let avg = total / (res * res) as f32;
        layer.iter_mut().for_each(|v| *v -= avg);
        let total_var: f32 = layer.iter().map(|x| x * x).sum();
        let stddev = f32::sqrt(total_var / (res * res) as f32);
        layer.iter_mut().for_each(|v| *v /= stddev * 3.)
    }

    let mut img = Vec::with_capacity((res * res) as usize);
    coords.array_chunks::<256>().for_each(|coords| {
        let output: Tensor2D<256, 3> = table.forward(&coords);
        let output = output.data().iter().flatten();
        img.extend(output.map(|v| (v * 256. - 0.5) as u8));
    });

    (
        values.map(|layer| {
            layer
                .into_iter()
                .map(|v| ((v + 1.) / 2. * 255.) as u8)
                .collect()
        }),
        img,
    )
}

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_name = std::env::args().nth(1).expect("no file name given");

    let mut table = HashTable::<16, { 1 << 14 }>::default();

    let mut rng = StdRng::seed_from_u64(0);
    table.reset_params(&mut rng);

    println!("{:?}", table.grid_sizes());

    let file = File::open(file_name).expect("failed to open file");

    let mut decoder = Decoder::new(file).expect("can not decode file");
    let DecodingResult::U8(pixels) = decoder.read_image().expect("file does not contain image") else {
        panic!("file is not 8 bit");
    };

    let image = ImageView::new(ImageInfo::rgb8(256, 256), &pixels);
    let window = create_window("image", Default::default())?;
    window.set_image("reference", image)?;

    let pixels: Box<[[f32; 3]; 256 * 256]> = pixels
        .chunks_exact(3)
        .map(|c| [0, 1, 2].map(|i| (c[i] as f32 + 0.5) / 256.))
        .collect::<Box<[[f32; 3]]>>()
        .try_into()
        .expect("image has the wrong size");

    let coords: Box<[[f32; 2]; 256 * 256]> = (0..256)
        .flat_map(|y| (0..256).map(move |x| [(x as f32 + 0.5) / 256., (y as f32 + 0.5) / 256.]))
        .collect::<Box<[[f32; 2]]>>()
        .try_into()
        .unwrap();

    let mut opt = Adam::new(AdamConfig {
        lr: 10f32.powi(-2),
        betas: [0.9, 0.99],
        eps: 10f32.powi(-15),
        weight_decay: None,
    });

    let start = Instant::now();

    loop {
        let mut img = Vec::with_capacity(256 * 256);
        let loss = coords
            .array_chunks::<{ 256 * 16 }>()
            .zip(pixels.array_chunks())
            .map(|(coords, pixels)| {
                let pixels = TensorCreator::new(*pixels);
                let output = table.forward(coords);
                let output_iter = output.data().iter().flatten();
                img.extend(output_iter.map(|v| (v * 256. - 0.5) as u8));
                mse_loss(output, pixels)
            })
            .fold(Tensor0D::new(0.).with_diff_tape(), |a, b| b + a);
        let loss = div_scalar(loss, 16.);
        println!("loss = {}", loss.data() * 256.);

        if loss.data() * 256. < 0.05 {
            println!("took: {:?}", start.elapsed());

            let mut index = 32;
            let mut res = 256;
            let (mut layers, mut img) = render_layer(&table, res);
            window.add_event_handler(move |mut window, event, _flow| {
                if let WindowEvent::KeyboardInput(event) = event {
                    if event.input.state == ElementState::Pressed {
                        if event.input.key_code == Some(VirtualKeyCode::Left) {
                            index += 33 - 1;
                        } else if event.input.key_code == Some(VirtualKeyCode::Right) {
                            index += 1;
                        } else if event.input.key_code == Some(VirtualKeyCode::Up) {
                            res *= 2;
                            let (l, i) = render_layer(&table, res);
                            layers = l;
                            img = i;
                        } else if event.input.key_code == Some(VirtualKeyCode::Down) {
                            res /= 2;
                            let (l, i) = render_layer(&table, res);
                            layers = l;
                            img = i;
                        }
                        index %= 33;
                        let image = if index == 32 {
                            ImageView::new(ImageInfo::rgb8(res, res), &img)
                        } else {
                            ImageView::new(ImageInfo::mono8(res, res), &*layers[index])
                        };
                        window.set_image("output", &image);
                    }
                }
            })?;

            break;
        }

        let grad = backward(loss);

        opt.update(&mut table, grad).expect("some unused gradients");

        let image = ImageView::new(ImageInfo::rgb8(256, 256), &img);
        window.set_image("output", image)?;
    }
    window.wait_until_destroyed()?;
    Ok(())
}

#[cfg(test)]
mod tests {

    use std::iter::repeat;

    use dfdx::{
        tensor::{tensor, Tensor1D},
        tensor_ops::SelectTo,
    };

    #[test]
    pub fn overflow() {
        let data = tensor([0., 1., 2.]);

        let indices: Box<[usize; 1 << 20]> = repeat(0)
            .take(1 << 20)
            .collect::<Box<[_]>>()
            .try_into()
            .unwrap();

        let res: Tensor1D<{ 1 << 20 }> = data.select(&*indices);
        drop(res)
    }
}
