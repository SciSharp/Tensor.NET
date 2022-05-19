using Tensornet.Common;

namespace Tensornet{
    public partial class Tensor<T>{
        public static Tensor<T> operator*(Tensor<T> lhs, Tensor<T> rhs){
            return InterElemOperation.Execute<T>(lhs, rhs, InterElemOperationType.Mul);
        }
        public static Tensor<T> operator*(Tensor<T> lhs, T rhs){
            return InterElemOperation.Execute<T>(lhs, (Tensor<T>)rhs, InterElemOperationType.Mul);
        }
        public static Tensor<T> operator*(T lhs, Tensor<T> rhs){
            return InterElemOperation.Execute<T>((Tensor<T>)lhs, rhs, InterElemOperationType.Mul);
        }
    }
}