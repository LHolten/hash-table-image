use std::mem::{transmute, MaybeUninit};

// pub fn map_boxed<F, T, O, const L: usize>(slice: &[T; L], f: F) -> Box<[O; L]>
// where
//     F: FnMut(&T) -> O,
// {
//     let mut target = Vec::with_capacity(L);
//     target.extend(slice.iter().map(f));
//     debug_assert_eq!(target.len(), slice.len());
//     target
//         .into_boxed_slice()
//         .try_into()
//         .unwrap_or_else(|_| unreachable!())
// }

pub fn init_mut<I, T, const L: usize>(
    arr: &mut MaybeUninit<[T; L]>,
    iter: impl IntoIterator<Item = I>,
    mut f: impl FnMut(I) -> T,
) {
    iter_mut(arr).zip(iter).for_each(|(t, v)| {
        t.write(f(v));
    })
}

pub fn iter_mut<T, const L: usize>(
    arr: &mut MaybeUninit<[T; L]>,
) -> impl Iterator<Item = &mut MaybeUninit<T>> {
    transmute_array_mut(arr).iter_mut()
}

pub fn transmute_array_mut<T, const L: usize>(
    arr: &mut MaybeUninit<[T; L]>,
) -> &mut [MaybeUninit<T>; L] {
    unsafe { transmute(arr) }
}

pub fn map_boxed2<I, J, G, F, O, const L: usize, const M: usize>(
    iter1: I,
    mut iter2: G,
    mut f: F,
) -> Box<[[O; M]; L]>
where
    I: IntoIterator,
    I::Item: Copy,
    G: FnMut(I::Item) -> J,
    J: IntoIterator,
    F: FnMut(I::Item, J::Item) -> O,
{
    let mut target = Box::new_uninit();
    iter_mut(&mut target)
        .zip(iter1)
        .for_each(|(t, a)| init_mut(t, iter2(a), |b| f(a, b)));
    unsafe { target.assume_init() }
}
