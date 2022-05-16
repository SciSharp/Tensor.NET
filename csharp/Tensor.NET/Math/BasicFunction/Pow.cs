using Tensornet.Common;

namespace Tensornet.Math{
    public static partial class MathT{
        public static Tensor<double> Pow(Tensor<double> inp, int y){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Pow(x, y));
        }
        public static Tensor<float> Pow(Tensor<float> inp, int y){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Pow(x, y));
        }
        public static Tensor<double> Pow(Tensor<long> inp, int y){
            return OnElemOperation.Execute<long, double>(inp, x => System.Math.Pow(x, y));
        }
        public static Tensor<double> Pow(Tensor<int> inp, int y){
            return OnElemOperation.Execute<int, double>(inp, x => System.Math.Pow(x, y));
        }
        public static Tensor<double> Pow(Tensor<double> inp, double y){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Pow(x, y));
        }
        public static Tensor<float> Pow(Tensor<float> inp, double y){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Pow(x, (float)y));
        }
        public static Tensor<double> Pow(Tensor<long> inp, double y){
            return OnElemOperation.Execute<long, double>(inp, x => System.Math.Pow(x, y));
        }
        public static Tensor<double> Pow(Tensor<int> inp, double y){
            return OnElemOperation.Execute<int, double>(inp, x => System.Math.Pow(x, y));
        }


        public static Tensor<double> Pow(int y, Tensor<double> inp){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Pow(y, x));
        }
        public static Tensor<float> Pow(int y, Tensor<float> inp){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Pow(y, x));
        }
        public static Tensor<double> Pow(int y, Tensor<long> inp){
            return OnElemOperation.Execute<long, double>(inp, x => System.Math.Pow(y, x));
        }
        public static Tensor<double> Pow(int y, Tensor<int> inp){
            return OnElemOperation.Execute<int, double>(inp, x => System.Math.Pow(y, x));
        }
        public static Tensor<double> Pow(double y, Tensor<double> inp){
            return OnElemOperation.Execute<double, double>(inp, x => System.Math.Pow(y, x));
        }
        public static Tensor<float> Pow(double y, Tensor<float> inp){
            return OnElemOperation.Execute<float, float>(inp, x => System.MathF.Pow((float)y, x));
        }
        public static Tensor<double> Pow(double y, Tensor<long> inp){
            return OnElemOperation.Execute<long, double>(inp, x => System.Math.Pow(y, x));
        }
        public static Tensor<double> Pow(double y, Tensor<int> inp){
            return OnElemOperation.Execute<int, double>(inp, x => System.Math.Pow(y, x));
        }
    }
}