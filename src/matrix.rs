use std::{
    iter::{repeat, repeat_with, zip},
    ops::{Index, IndexMut},
};

use crate::Point;

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

pub fn point_to_index<const D: usize>(point: Point<D>, dimension_offsets: &[usize; D]) -> usize {
    zip(point[1..D].into_iter(), dimension_offsets.into_iter())
        .map(|(coordinate, dimension_offset)| coordinate * dimension_offset)
        .sum::<usize>()
        + point[0]
}

pub fn index_to_point<const D: usize>(
    mut index: usize,
    dimension_offsets: &[usize; D],
) -> Point<D> {
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

#[derive(Debug)]
pub struct Matrix<T, const D: usize> {
    dimensions: [usize; D],
    dimension_offsets: [usize; D],
    matrix: Vec<T>,
}

impl<T, const D: usize> Matrix<T, D>
where
    T: Copy,
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
    pub fn matrix(&self) -> &Vec<T> {
        &self.matrix
    }

    #[inline(always)]
    pub fn matrix_mut(&mut self) -> &mut Vec<T> {
        &mut self.matrix
    }

    #[inline(always)]
    pub fn get(&self, point: Point<D>) -> Option<&T> {
        self.matrix
            .get(point_to_index(point, &self.dimension_offsets))
    }

    #[inline(always)]
    pub fn get_mut(&mut self, point: Point<D>) -> Option<&mut T> {
        self.matrix
            .get_mut(point_to_index(point, &self.dimension_offsets))
    }
}

impl<T, const D: usize> Index<Point<D>> for Matrix<T, D> {
    type Output = <Self as Index<usize>>::Output;

    #[inline(always)]
    fn index(&self, index: Point<D>) -> &Self::Output {
        &self[point_to_index(index, &self.dimension_offsets)]
    }
}

impl<I, T, const D: usize> Index<I> for Matrix<T, D>
where
    Vec<T>: Index<I>,
{
    type Output = <Vec<T> as Index<I>>::Output;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        &self.matrix[index]
    }
}

impl<T, const D: usize> IndexMut<Point<D>> for Matrix<T, D> {
    #[inline(always)]
    fn index_mut(&mut self, index: Point<D>) -> &mut Self::Output {
        let index = point_to_index(index, &self.dimension_offsets);
        &mut self[index]
    }
}

impl<I, T, const D: usize> IndexMut<I> for Matrix<T, D>
where
    Vec<T>: IndexMut<I>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.matrix[index]
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
    fn point_to_index() {
        let index = super::point_to_index(Point::from([1, 2]), &[4, 20]);

        assert_eq!(index, 9);
    }

    #[test]
    fn index_to_point() {
        let point = super::index_to_point(9, &[4, 20]);

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
        assert_eq!(matrix.get(Point::from([4, 5])), None);
    }

    #[test]
    fn get_mut() {
        let mut matrix = Matrix {
            dimensions: [4, 5],
            dimension_offsets: [4, 20],
            matrix: (0..20).collect(),
        };

        assert_eq!(matrix.get_mut(Point::from([1, 2])), Some(&mut 9));
        assert_eq!(matrix.get_mut(Point::from([4, 5])), None);
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
}
