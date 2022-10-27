//using NN.Native.Abstraction.DType;

//namespace NN.Native.Basic.DType
//{
//    public class Int32Handler : IUnaryDTypeHandler<int>, IBinaryDTypeHandler<int, int>, ITernaryDTypeHandler<int, int, int>, INativeConvertible<int, int>, INativeConvertible<bool, int>
//    {
//        #region unary
//        public static int Zero { get => 0; }
//        public static int One { get => 1; }
//        #endregion

//        #region binary
//        public static void AddInplace(ref int src, in int addend) => src += addend;
//        public static void SubstractInplace(ref int src, in int subtrahend) => src -= subtrahend;
//        public static void MultiplyInplace(ref int src, in int multiplicator) => src *= multiplicator;
//        public static void DivideInplace(ref int src, in int divisor) => src /= divisor;
//        public static void ModuleInplace(ref int src, in int modulus) => src %= modulus;
//        public static void LeftShift(in int src, ref int dst, int offset) => dst = src << offset;
//        public static void RightShift(in int src, ref int dst, int offset) => dst = src >> offset;
//        public static bool AreEqual(in int a, in int b) => a == b;
//        public static bool IsGraeter(in int a, in int b) => a > b;
//        public static bool IsLess(in int a, in int b) => a < b;
//        #endregion

//        #region ternary
//        public static void Add(in int a, in int b, ref int dst) => dst = a + b;
//        public static void Substract(in int a, in int b, ref int dst) => dst = a - b;
//        public static void Multiply(in int a, in int b, ref int dst) => dst = a * b;
//        public static void Divide(in int a, in int b, ref int dst) => dst = a / b;
//        public static void Module(in int a, in int b, ref int dst) => dst = a % b;
//        public static void MultiplyAndAdd(in int a, in int b, ref int dst) => dst += a * b;
//        #endregion

//        #region convert
//        public static void Convert(in int a, ref int b) => b = a;
//        public static void Convert(in bool a, ref int b) => b = a ? 1 : 0;
//        #endregion
//    }
//}
