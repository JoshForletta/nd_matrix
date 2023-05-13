use std::{
    array::from_fn,
    iter::{once, repeat, repeat_with, zip},
    ops::{Index, IndexMut},
    slice::{Iter, IterMut},
    vec::IntoIter,
};

use crate::{AxisPair, Point};

pub fn dimension_offsets<const D: usize>(dimensions: &[usize; D]) -> [usize; D] {
    let mut dimension_offsets = [0; D];
    let mut cumulative_product = 1;

    for (dimension_offset, dimension_size) in
        zip((&mut dimension_offsets).into_iter(), dimensions.into_iter())
    {
        cumulative_product = cumulative_product * dimension_size;
        *dimension_offset = cumulative_product;
    }

    dimension_offsets
}

pub trait ToIndex<const D: usize> {
    fn to_index(&self, dimension_offsets: &[usize; D]) -> usize;
}

impl<const D: usize> ToIndex<D> for Point<D> {
    fn to_index(&self, dimension_offsets: &[usize; D]) -> usize {
        zip(self[1..D].into_iter(), dimension_offsets.into_iter())
            .map(|(coordinate, dimension_offset)| coordinate * dimension_offset)
            .sum::<usize>()
            + self[0]
    }
}

impl<const D: usize> ToIndex<D> for usize {
    #[inline(always)]
    fn to_index(&self, _dimension_offsets: &[usize; D]) -> usize {
        *self
    }
}

pub trait ToPoint<const D: usize> {
    fn to_point(&self, dimension_offsets: &[usize; D]) -> Point<D>;
}

impl<const D: usize> ToPoint<D> for Point<D> {
    #[inline(always)]
    fn to_point(&self, _dimension_offsets: &[usize; D]) -> Point<D> {
        *self
    }
}

impl<const D: usize> ToPoint<D> for usize {
    fn to_point(&self, dimension_offsets: &[usize; D]) -> Point<D> {
        let mut index = self.clone();
        let mut point = Point::from([0; D]);

        for (coordinate, dimension_offset) in
            zip(point[1..D].iter_mut(), dimension_offsets.iter()).rev()
        {
            *coordinate = index / dimension_offset;
            index = index % dimension_offset;
        }

        point[0] = index;

        point
    }
}

#[derive(Debug)]
pub struct Matrix<T, const D: usize> {
    dimensions: [usize; D],
    dimension_offsets: [usize; D],
    matrix: Vec<T>,
}

impl<T, const D: usize> Matrix<T, D>
where
    T: Clone,
{
    pub fn fill(dimensions: [usize; D], cell: T) -> Self {
        let capacity = dimensions.iter().product();

        Self {
            dimensions,
            dimension_offsets: dimension_offsets(&dimensions),
            matrix: repeat(cell).take(capacity).collect(),
        }
    }
}

impl<T, const D: usize> Matrix<T, D> {
    pub fn fill_with<F: FnMut() -> T>(dimensions: [usize; D], cell: F) -> Self {
        let capacity = dimensions.iter().product();

        Self {
            dimensions,
            dimension_offsets: dimension_offsets(&dimensions),
            matrix: repeat_with(cell).take(capacity).collect(),
        }
    }

    #[inline(always)]
    pub fn dimensions(&self) -> &[usize; D] {
        &self.dimensions
    }

    #[inline(always)]
    pub fn dimension_offsets(&self) -> &[usize; D] {
        &self.dimension_offsets
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.matrix.len()
    }

    #[inline(always)]
    pub fn matrix(&self) -> &Vec<T> {
        &self.matrix
    }

    #[inline(always)]
    pub fn matrix_mut(&mut self) -> &mut Vec<T> {
        &mut self.matrix
    }

    #[inline(always)]
    pub fn get<I>(&self, index: I) -> Option<&T>
    where
        I: ToIndex<D>,
    {
        self.matrix.get(index.to_index(&self.dimension_offsets))
    }

    #[inline(always)]
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut T>
    where
        I: ToIndex<D>,
    {
        self.matrix.get_mut(index.to_index(&self.dimension_offsets))
    }

    pub fn get_adjacent_indexes<I>(&self, index: I) -> [AxisPair<Option<usize>>; D]
    where
        I: ToIndex<D>,
    {
        let index = index.to_index(&self.dimension_offsets);

        let mut adjacent_indexes: [AxisPair<Option<usize>>; D] = [Default::default(); D];
        let coordinate_offsets: Vec<_> = once(&1).chain(self.dimension_offsets.iter()).collect();

        for dimension in 0..D {
            let corrdinate_offset = *coordinate_offsets[dimension];

            let dimension_offset = self.dimension_offsets[dimension];
            let higher_dimension_index = index / dimension_offset;
            let lower_bound = higher_dimension_index * dimension_offset;
            let upper_bound = (higher_dimension_index + 1) * dimension_offset;

            adjacent_indexes[dimension].pos = index
                .checked_add(corrdinate_offset)
                .filter(|index| index < &upper_bound);

            adjacent_indexes[dimension].neg = index
                .checked_sub(corrdinate_offset)
                .filter(|index| index >= &lower_bound);
        }

        adjacent_indexes
    }

    pub fn get_adjacencies<I>(&self, index: I) -> [AxisPair<Option<&T>>; D]
    where
        I: ToIndex<D>,
    {
        let adjacent_index_pairs = self.get_adjacent_indexes(index);

        let mut adjacencies: [AxisPair<Option<&T>>; D] = from_fn(|_| AxisPair::default());

        for dimension in 0..D {
            if let Some(adjacent_index) = adjacent_index_pairs[dimension].pos {
                adjacencies[dimension].pos = Some(&self.matrix[adjacent_index]);
            }

            if let Some(adjacent_index) = adjacent_index_pairs[dimension].neg {
                adjacencies[dimension].neg = Some(&self.matrix[adjacent_index]);
            }
        }

        adjacencies
    }
}

impl<I, T, const D: usize> Index<I> for Matrix<T, D>
where
    I: ToIndex<D>,
{
    type Output = <Vec<T> as Index<usize>>::Output;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        &self.matrix[index.to_index(&self.dimension_offsets)]
    }
}

impl<I, T, const D: usize> IndexMut<I> for Matrix<T, D>
where
    I: ToIndex<D>,
    Vec<T>: IndexMut<usize>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let index = index.to_index(&self.dimension_offsets);
        &mut self.matrix[index]
    }
}

impl<'a, T, const D: usize> IntoIterator for Matrix<T, D> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.matrix.into_iter()
    }
}

impl<'a, T, const D: usize> IntoIterator for &'a Matrix<T, D> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        (&self.matrix).into_iter()
    }
}

impl<'a, T, const D: usize> IntoIterator for &'a mut Matrix<T, D> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.matrix).into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_offsets() {
        let dimension_offsets = super::dimension_offsets(&[4, 5]);

        assert_eq!(dimension_offsets, [4, 20]);
    }

    #[test]
    fn to_index() {
        let index = Point::from([1, 2]).to_index(&[4, 20]);

        assert_eq!(index, 9);
    }

    #[test]
    fn to_point() {
        let point = 9.to_point(&[4, 20]);

        assert_eq!(point, Point::from([1, 2]));
    }

    #[test]
    fn fill() {
        let matrix = Matrix::fill([4, 5], 0);

        assert_eq!(matrix.matrix, Vec::from([0; 20]));
    }

    #[test]
    fn fill_with() {
        let matrix = Matrix::fill_with([4, 5], || 0);

        assert_eq!(matrix.matrix, Vec::from([0; 20]));
    }

    #[test]
    fn dimensions_getter() {
        let matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: Vec::from([0; 20]),
        };

        assert_eq!(matrix.dimensions(), &[4, 5]);
    }

    #[test]
    fn dimension_offsets_getter() {
        let matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: Vec::from([0; 20]),
        };

        assert_eq!(matrix.dimension_offsets(), &[4, 20]);
    }

    #[test]
    fn matrix_getter() {
        let matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: Vec::from([0; 20]),
        };

        assert_eq!(matrix.matrix(), &Vec::from([0; 20]));
    }

    #[test]
    fn matrix_mut_getter() {
        let mut matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: Vec::from([0; 20]),
        };

        assert_eq!(matrix.matrix_mut(), &mut Vec::from([0; 20]));
    }

    #[test]
    fn get() {
        let matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: (0..20).collect(),
        };

        assert_eq!(matrix.get(Point::from([1, 2])), Some(&9));
        assert_eq!(matrix.get(9), Some(&9));
        assert_eq!(matrix.get(Point::from([4, 5])), None);
        assert_eq!(matrix.get(20), None);
    }

    #[test]
    fn get_mut() {
        let mut matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: (0..20).collect(),
        };

        assert_eq!(matrix.get_mut(Point::from([1, 2])), Some(&mut 9));
        assert_eq!(matrix.get_mut(9), Some(&mut 9));
        assert_eq!(matrix.get_mut(Point::from([4, 5])), None);
        assert_eq!(matrix.get_mut(20), None);
    }

    #[test]
    fn index_usize() {
        let matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: (0..20).collect(),
        };

        assert_eq!(matrix[9], 9);
    }

    #[test]
    #[should_panic]
    fn index_usize_out_of_bounds() {
        let matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: (0..20).collect(),
        };

        matrix[20];
    }

    #[test]
    fn index_point() {
        let matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: (0..20).collect(),
        };

        assert_eq!(matrix[Point::from([1, 2])], 9);
    }

    #[test]
    #[should_panic]
    fn index_point_out_of_bounds() {
        let matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: (0..20).collect(),
        };

        matrix[Point::from([4, 5])];
    }

    #[test]
    fn index_usize_mut() {
        let mut matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: (0..20).collect(),
        };

        assert_eq!(&mut matrix[9], &mut 9);
    }

    #[test]
    #[should_panic]
    fn index_usize_mut_out_of_bounds() {
        let mut matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: (0..20).collect(),
        };

        (&mut matrix)[20];
    }

    #[test]
    fn index_point_mut() {
        let mut matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: (0..20).collect(),
        };

        assert_eq!(&mut matrix[Point::from([1, 2])], &mut 9);
    }

    #[test]
    #[should_panic]
    fn index_point_mut_out_of_bounds() {
        let mut matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: (0..20).collect(),
        };

        (&mut matrix)[Point::from([4, 5])];
    }

    #[test]
    fn into_iter() {
        let matrix = Matrix {
            dimensions: [2, 2],
            dimension_offsets: [2, 4],
            matrix: (0..4).collect(),
        };

        let mut iter = matrix.into_iter();

        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn into_iter_ref() {
        let matrix = Matrix {
            dimensions: [2, 2],
            dimension_offsets: [2, 4],
            matrix: (0..4).collect(),
        };

        let mut iter = (&matrix).into_iter();

        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn into_iter_mut_ref() {
        let mut matrix = Matrix {
            dimensions: [2, 2],
            dimension_offsets: [2, 4],
            matrix: (0..4).collect(),
        };

        let mut iter = (&mut matrix).into_iter();

        assert_eq!(iter.next(), Some(&mut 0));
        assert_eq!(iter.next(), Some(&mut 1));
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next(), Some(&mut 3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn get_adjacent_indexes() {
        let matrix = Matrix {
            dimensions: [2, 2],
            dimension_offsets: [2, 4],
            matrix: (1..=4).collect(),
        };

        assert_eq!(
            matrix.get_adjacent_indexes(0),
            [AxisPair::new(Some(1), None), AxisPair::new(Some(2), None)]
        );

        assert_eq!(
            matrix.get_adjacent_indexes(1),
            [AxisPair::new(None, Some(0)), AxisPair::new(Some(3), None)]
        );

        assert_eq!(
            matrix.get_adjacent_indexes(2),
            [AxisPair::new(Some(3), None), AxisPair::new(None, Some(0))]
        );

        assert_eq!(
            matrix.get_adjacent_indexes(3),
            [AxisPair::new(None, Some(2)), AxisPair::new(None, Some(1))]
        );
    }

    #[test]
    fn get_adjacencies() {
        let matrix = Matrix {
            dimensions: [2, 2],
            dimension_offsets: [2, 4],
            matrix: (1..=4).collect(),
        };

        assert_eq!(
            matrix.get_adjacencies(0),
            [AxisPair::new(Some(&2), None), AxisPair::new(Some(&3), None)]
        );

        assert_eq!(
            matrix.get_adjacencies(1),
            [AxisPair::new(None, Some(&1)), AxisPair::new(Some(&4), None)]
        );

        assert_eq!(
            matrix.get_adjacencies(2),
            [AxisPair::new(Some(&4), None), AxisPair::new(None, Some(&1))]
        );

        assert_eq!(
            matrix.get_adjacencies(3),
            [AxisPair::new(None, Some(&3)), AxisPair::new(None, Some(&2))]
        );
    }
}
