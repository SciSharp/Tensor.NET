using Numnet.Math;
using Numnet.Common;
using Numnet.Native;

namespace Numnet{
    public partial class Tensor<T> where T : struct, IEquatable<T>, IConvertible{
        public static Tensor<bool> operator==(Tensor<T> lhs, Tensor<double> rhs){
            return InterElemOperation.Execute<T, double, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) == y);
        }
        public static Tensor<bool> operator!=(Tensor<T> lhs, Tensor<double> rhs){
            return InterElemOperation.Execute<T, double, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) != y);
        }
        public static Tensor<bool> operator==(Tensor<T> lhs, Tensor<float> rhs){
            return InterElemOperation.Execute<T, float, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) == y);
        }
        public static Tensor<bool> operator!=(Tensor<T> lhs, Tensor<float> rhs){
            return InterElemOperation.Execute<T, float, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) != y);
        }
        public static Tensor<bool> operator==(Tensor<T> lhs, Tensor<long> rhs){
            return InterElemOperation.Execute<T, long, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) == y);
        }
        public static Tensor<bool> operator!=(Tensor<T> lhs, Tensor<long> rhs){
            return InterElemOperation.Execute<T, long, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) != y);
        }
        public static Tensor<bool> operator==(Tensor<T> lhs, Tensor<int> rhs){
            return InterElemOperation.Execute<T, int, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) == y);
        }
        public static Tensor<bool> operator!=(Tensor<T> lhs, Tensor<int> rhs){
            return InterElemOperation.Execute<T, int, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) != y);
        }
        public static Tensor<bool> operator==(Tensor<T> lhs, Tensor<bool> rhs){
            return InterElemOperation.Execute<T, bool, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) == (y ? 1 : 0));
        }
        public static Tensor<bool> operator!=(Tensor<T> lhs, Tensor<bool> rhs){
            return InterElemOperation.Execute<T, bool, bool>(lhs, rhs, (x, y) => Convert.ToDouble(x) != (y ? 1 : 0));
        }

        public override bool Equals(object? obj)
        {
            if (obj is null || GetType() != obj.GetType())
            {
                return false;
            }
            if (ReferenceEquals(this, obj)) return true;
            return Equals((Tensor<T>) obj);
        }

        public override int GetHashCode()
        {
            int valueHash = DataType switch
            {
                DType.Bool => TMemory.GetHashCode() % 0x3f3f3f3f,
                DType.Int32 => TMemory.GetHashCode() % 0x3f3f3f3f,
                DType.Int64 => TMemory.GetHashCode() % 0x3f3f3f3f,
                DType.Float32 => TMemory.GetHashCode() % 0x3f3f3f3f,
                DType.Float64 => TMemory.GetHashCode() % 0x3f3f3f3f,
                _ => throw new NotImplementedException()
            };
            return (valueHash + (int)DataType * 0x0cfcfcfc) % 0x3f3f3f3f;
        }
    }
}