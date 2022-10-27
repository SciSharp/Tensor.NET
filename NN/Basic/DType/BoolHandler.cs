//using NN.Native.Abstraction.DType;

//namespace NN.Native.Basic.DType
//{
//    public class boolHandler : IUnaryDTypeHandler<bool>, IBinaryDTypeHandler<bool, bool>, ITernaryDTypeHandler<bool, bool, bool>, INativeConvertible<bool, bool>
//    {
//        #region unary
//        public static bool Zero { get => false; }
//        public static bool One { get => true; }
//        #endregion

//        #region binary
//        public static void AddInplace(ref bool src, in bool addend) => throw new InvalidOperationException();
//        public static void SubstractInplace(ref bool src, in bool subtrahend) => throw new InvalidOperationException();
//        public static void MultiplyInplace(ref bool src, in bool multiplicator) => throw new InvalidOperationException();
//        public static void DivideInplace(ref bool src, in bool divisor) => throw new InvalidOperationException();
//        public static void ModuleInplace(ref bool src, in bool modulus) => throw new InvalidOperationException();
//        //public static void LeftShift(ref bool src, ref bool dst, uint offset) => throw new InvalidOperationException();
//        //public static void RightShift(ref bool src, ref bool dst, uint offset) => throw new InvalidOperationException();
//        public static bool AreEqual(in bool a, in bool b) => a == b;
//        public static bool IsGraeter(in bool a, in bool b) => throw new InvalidOperationException();
//        public static bool IsLess(in bool a, in bool b) => throw new InvalidOperationException();
//        #endregion

//        #region ternary
//        public static void Add(in bool a, in bool b, ref bool dst) => throw new InvalidOperationException();
//        public static void Substract(in bool a, in bool b, ref bool dst) => throw new InvalidOperationException();
//        public static void Multiply(in bool a, in bool b, ref bool dst) => throw new InvalidOperationException();
//        public static void Divide(in bool a, in bool b, ref bool dst) => throw new InvalidOperationException();
//        public static void Module(in bool a, in bool b, ref bool dst) => throw new InvalidOperationException();
//        public static void MultiplyAndAdd(in bool a, in bool b, ref bool dst) => throw new InvalidOperationException();
//        #endregion

//        #region convert
//        public static void Convert(in bool a, ref bool b) => b = a;
//        #endregion
//    }
//}
