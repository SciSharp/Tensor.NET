using Numnet.Math;
using Numnet.Common;
using Numnet.Native;

namespace Numnet{
    public partial class Tensor<T> where T : struct, IEquatable<T>, IConvertible{
        public static Tensor<bool> operator<(Tensor<T> lhs, Tensor<double> rhs){
            return InterElemOperation.Execute<T, double, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) < y);
        }
        public static Tensor<bool> operator<=(Tensor<T> lhs, Tensor<double> rhs){
            return InterElemOperation.Execute<T, double, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) <= y);
        }
        public static Tensor<bool> operator<(Tensor<T> lhs, Tensor<float> rhs){
            return InterElemOperation.Execute<T, float, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) < y);
        }
        public static Tensor<bool> operator<=(Tensor<T> lhs, Tensor<float> rhs){
            return InterElemOperation.Execute<T, float, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) <= y);
        }
        public static Tensor<bool> operator<(Tensor<T> lhs, Tensor<long> rhs){
            return InterElemOperation.Execute<T, long, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) < y);
        }
        public static Tensor<bool> operator<=(Tensor<T> lhs, Tensor<long> rhs){
            return InterElemOperation.Execute<T, long, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) <= y);
        }
        public static Tensor<bool> operator<(Tensor<T> lhs, Tensor<int> rhs){
            return InterElemOperation.Execute<T, int, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) < y);
        }
        public static Tensor<bool> operator<=(Tensor<T> lhs, Tensor<int> rhs){
            return InterElemOperation.Execute<T, int, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) <= y);
        }
        public static Tensor<bool> operator<(Tensor<T> lhs, Tensor<bool> rhs){
            return InterElemOperation.Execute<T, bool, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) < (y ? 1 : 0));
        }
        public static Tensor<bool> operator<=(Tensor<T> lhs, Tensor<bool> rhs){
            return InterElemOperation.Execute<T, bool, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) <= (y ? 1 : 0));
        }
    }
}