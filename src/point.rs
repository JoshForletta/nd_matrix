use std::{
    array::IntoIter,
    ops::{Index, IndexMut},
    slice::{Iter, IterMut},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point<const N: usize>([usize; N]);

impl<const D: usize> Point<D> {
    #[inline(always)]
    pub fn coordinates(&self) -> &[usize; D] {
        &self.0
    }

    #[inline(always)]
    pub fn coordinates_mut(&mut self) -> &mut [usize; D] {
        &mut self.0
    }
}

impl<const D: usize> From<[usize; D]> for Point<D> {
    fn from(point: [usize; D]) -> Self {
        Point(point)
    }
}

impl<const D: usize> IntoIterator for Point<D> {
    type Item = usize;
    type IntoIter = IntoIter<usize, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, const D: usize> IntoIterator for &'a Point<D> {
    type Item = &'a usize;
    type IntoIter = Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        (&self.0).into_iter()
    }
}

impl<'a, const D: usize> IntoIterator for &'a mut Point<D> {
    type Item = &'a mut usize;
    type IntoIter = IterMut<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.0).into_iter()
    }
}

impl<I, const D: usize> Index<I> for Point<D>
where
    [usize]: Index<I>,
{
    type Output = <[usize] as Index<I>>::Output;

    #[inline(always)]
    fn index(&self, index: I) -> &<Self as Index<I>>::Output {
        self.0.index(index)
    }
}

impl<I, const D: usize> IndexMut<I> for Point<D>
where
    [usize]: IndexMut<I>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut <Self as Index<I>>::Output {
        self.0.index_mut(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinates() {
        let point = Point([1, 2]);

        assert_eq!(point.coordinates(), &[1, 2]);
    }

    #[test]
    fn coordinates_mut() {
        let mut point = Point([1, 2]);

        assert_eq!(point.coordinates_mut(), &mut [1, 2]);
    }

    #[test]
    fn from() {
        let point = Point::from([1, 2]);

        assert_eq!(point, Point([1, 2]));
    }

    #[test]
    fn into_iter() {
        let point = Point([1, 2]);
        let mut iter = point.into_iter();

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn into_iter_ref() {
        let point = Point([1, 2]);
        let mut iter = (&point).into_iter();

        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn into_iter_mut_ref() {
        let mut point = Point([1, 2]);
        let mut iter = (&mut point).into_iter();

        assert_eq!(iter.next(), Some(&mut 1));
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn index() {
        let point = Point([1, 2]);

        assert_eq!(point[0], 1);
        assert_eq!(point[1], 2);
    }

    #[test]
    fn index_mut() {
        let mut point = Point([1, 2]);

        assert_eq!(&mut point[0], &mut 1);
        assert_eq!(&mut point[1], &mut 2);
    }
}
