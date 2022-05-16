using Tensornet.Common;

namespace Tensornet.Math{
    public static partial class MathT{
        public static Tensor<double> Min(this Tensor<double> a, Tensor<double> b){
            return InterElemOperation.Execute<double, double, double>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<double> Min(this Tensor<double> a, Tensor<float> b){
            return InterElemOperation.Execute<double, float, double>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<double> Min(this Tensor<double> a, Tensor<long> b){
            return InterElemOperation.Execute<double, long, double>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<double> Min(this Tensor<double> a, Tensor<int> b){
            return InterElemOperation.Execute<double, int, double>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<double> Min(this Tensor<double> a, Tensor<bool> b){
            return InterElemOperation.Execute<double, bool, double>(a, b, (x, y) => y ? (x > 1 ? 1 : x) : (x > 0 ? 0 : x));
        }
        public static Tensor<double> Min(this Tensor<float> a, Tensor<double> b){
            return InterElemOperation.Execute<float, double, double>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<float> Min(this Tensor<float> a, Tensor<float> b){
            return InterElemOperation.Execute<float, float, float>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<float> Min(this Tensor<float> a, Tensor<long> b){
            return InterElemOperation.Execute<float, long, float>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<float> Min(this Tensor<float> a, Tensor<int> b){
            return InterElemOperation.Execute<float, int, float>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<float> Min(this Tensor<float> a, Tensor<bool> b){
            return InterElemOperation.Execute<float, bool, float>(a, b, (x, y) => y ? (x > 1 ? 1 : x) : (x > 0 ? 0 : x));
        }
        public static Tensor<double> Min(this Tensor<long> a, Tensor<double> b){
            return InterElemOperation.Execute<long, double, double>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<float> Min(this Tensor<long> a, Tensor<float> b){
            return InterElemOperation.Execute<long, float, float>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<long> Min(this Tensor<long> a, Tensor<long> b){
            return InterElemOperation.Execute<long, long, long>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<long> Min(this Tensor<long> a, Tensor<int> b){
            return InterElemOperation.Execute<long, int, long>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<long> Min(this Tensor<long> a, Tensor<bool> b){
            return InterElemOperation.Execute<long, bool, long>(a, b, (x, y) => y ? (x > 1 ? 1 : x) : (x > 0 ? 0 : x));
        }
        public static Tensor<double> Min(this Tensor<int> a, Tensor<double> b){
            return InterElemOperation.Execute<int, double, double>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<float> Min(this Tensor<int> a, Tensor<float> b){
            return InterElemOperation.Execute<int, float, float>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<long> Min(this Tensor<int> a, Tensor<long> b){
            return InterElemOperation.Execute<int, long, long>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<int> Min(this Tensor<int> a, Tensor<int> b){
            return InterElemOperation.Execute<int, int, int>(a, b, (x, y) => x > y ? y : x);
        }
        public static Tensor<int> Min(this Tensor<int> a, Tensor<bool> b){
            return InterElemOperation.Execute<int, bool, int>(a, b, (x, y) => y ? (x > 1 ? 1 : x) : (x > 0 ? 0 : x));
        }
        public static Tensor<double> Min(this Tensor<bool> a, Tensor<double> b){
            return InterElemOperation.Execute<bool, double, double>(a, b, (x, y) => x ? (y > 1 ? 1 : y) : (y > 0 ? 0 : y));
        }
        public static Tensor<float> Min(this Tensor<bool> a, Tensor<float> b){
            return InterElemOperation.Execute<bool, float, float>(a, b, (x, y) => x ? (y > 1 ? 1 : y) : (y > 0 ? 0 : y));
        }
        public static Tensor<long> Min(this Tensor<bool> a, Tensor<long> b){
            return InterElemOperation.Execute<bool, long, long>(a, b, (x, y) => x ? (y > 1 ? 1 : y) : (y > 0 ? 0 : y));
        }
        public static Tensor<int> Min(this Tensor<bool> a, Tensor<int> b){
            return InterElemOperation.Execute<bool, int, int>(a, b, (x, y) => x ? (y > 1 ? 1 : y) : (y > 0 ? 0 : y));
        }
        public static Tensor<bool> Min(this Tensor<bool> a, Tensor<bool> b){
            return InterElemOperation.Execute<bool, bool, bool>(a, b, (x, y) => x && y);
        }
    }
}