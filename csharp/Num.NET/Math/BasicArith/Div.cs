using Numnet.Common;

namespace Numnet.Math{
    public static class DivExtension{
        public static Tensor<double> Div(this Tensor<double> a, Tensor<double> b){
            return InterElemOperation.Execute<double, double, double>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<double> Div(this Tensor<double> a, Tensor<float> b){
            return InterElemOperation.Execute<double, float, double>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<double> Div(this Tensor<double> a, Tensor<long> b){
            return InterElemOperation.Execute<double, long, double>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<double> Div(this Tensor<double> a, Tensor<int> b){
            return InterElemOperation.Execute<double, int, double>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<double> Div(this Tensor<double> a, Tensor<bool> b){
            return InterElemOperation.Execute<double, bool, double>(a, b, (x, y) => y ? x : Double.NaN);
        }
                            
        public static Tensor<double> Div(this Tensor<float> a, Tensor<double> b){
            return InterElemOperation.Execute<float, double, double>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<float> Div(this Tensor<float> a, Tensor<float> b){
            return InterElemOperation.Execute<float, float, float>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<float> Div(this Tensor<float> a, Tensor<long> b){
            return InterElemOperation.Execute<float, long, float>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<float> Div(this Tensor<float> a, Tensor<int> b){
            return InterElemOperation.Execute<float, int, float>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<float> Div(this Tensor<float> a, Tensor<bool> b){
            return InterElemOperation.Execute<float, bool, float>(a, b, (x, y) => y ? x : Single.NaN);
        }
                            
        public static Tensor<double> Div(this Tensor<long> a, Tensor<double> b){
            return InterElemOperation.Execute<long, double, double>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<float> Div(this Tensor<long> a, Tensor<float> b){
            return InterElemOperation.Execute<long, float, float>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<double> Div(this Tensor<long> a, Tensor<long> b){
            return InterElemOperation.Execute<long, long, double>(a, b, (x, y) => (double)x / (double)y);
        }
                            
        public static Tensor<double> Div(this Tensor<long> a, Tensor<int> b){
            return InterElemOperation.Execute<long, int, double>(a, b, (x, y) => (double)x / (double)y);
        }
                            
        public static Tensor<double> Div(this Tensor<long> a, Tensor<bool> b){
            return InterElemOperation.Execute<long, bool, double>(a, b, (x, y) => y ? x : Double.NaN);
        }
                            
        public static Tensor<double> Div(this Tensor<int> a, Tensor<double> b){
            return InterElemOperation.Execute<int, double, double>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<float> Div(this Tensor<int> a, Tensor<float> b){
            return InterElemOperation.Execute<int, float, float>(a, b, (x, y) => x / y);
        }
                            
        public static Tensor<double> Div(this Tensor<int> a, Tensor<long> b){
            return InterElemOperation.Execute<int, long, double>(a, b, (x, y) => (double)x / (double)y);
        }
                            
        public static Tensor<double> Div(this Tensor<int> a, Tensor<int> b){
            return InterElemOperation.Execute<int, int, double>(a, b, (x, y) => (double)x / (double)y);
        }
                            
        public static Tensor<double> Div(this Tensor<int> a, Tensor<bool> b){
            return InterElemOperation.Execute<int, bool, double>(a, b, (x, y) => y ? x : Double.NaN);
        }
                            
        public static Tensor<double> Div(this Tensor<bool> a, Tensor<double> b){
            return InterElemOperation.Execute<bool, double, double>(a, b, (x, y) => x ? 1 / y : 0);
        }
                            
        public static Tensor<float> Div(this Tensor<bool> a, Tensor<float> b){
            return InterElemOperation.Execute<bool, float, float>(a, b, (x, y) => x ? 1 / y : 0);
        }
                            
        public static Tensor<double> Div(this Tensor<bool> a, Tensor<long> b){
            return InterElemOperation.Execute<bool, long, double>(a, b, (x, y) => x ? 1 / (double)y : 0);
        }
                            
        public static Tensor<double> Div(this Tensor<bool> a, Tensor<int> b){
            return InterElemOperation.Execute<bool, int, double>(a, b, (x, y) => x ? 1 / (double)y : 0);
        }
                            
        public static Tensor<double> Div(this Tensor<bool> a, Tensor<bool> b){
            return InterElemOperation.Execute<bool, bool, double>(a, b, (x, y) => y ? (x ? 1 : 0) : Double.NaN);
        }
    }
}