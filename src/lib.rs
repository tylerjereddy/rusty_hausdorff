//extern crate num_traits;
//use num_traits::Num;

pub fn directed_hausdorff(ar1: &[f64], ar2: &[f64]) -> f64
{
    
//	let mut cmax = 0.0;
    //let inf = f64::INFINITY;
    let mut d = 0.0;

	for &item_i in ar1.iter() {
        for &item_j in ar2.iter() {
            d += &item_i + item_j;
}
}
d
}



#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let a1 = vec![1.0, 2.0, 3.0];
        let a2 = vec![1.0, 2.0, 3.0];
        assert_eq!(directed_hausdorff(&a1, &a2), 36.0);
    }
}
