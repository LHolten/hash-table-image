pub fn map_boxed<F, T, O, const L: usize>(slice: &[T; L], f: F) -> Box<[O; L]>
where
    F: FnMut(&T) -> O,
{
    let mut target = Vec::with_capacity(L);
    target.extend(slice.iter().map(f));
    debug_assert_eq!(target.len(), slice.len());
    unsafe { target.into_boxed_slice().try_into().unwrap_unchecked() }
}

pub fn map_inplace<F, T: Copy, const L: usize>(slice: &mut [T; L], mut f: F)
where
    F: FnMut(T) -> T,
{
    slice.iter_mut().for_each(|x| *x = f(*x));
}

// pub fn boxed_chunks<I: IntoIterator, const L: usize>(
//     iter: I,
// ) -> impl Iterator<Item = Box<I::Item, L>> {
//     let mut iter = iter.into_iter();
//     let
// }
