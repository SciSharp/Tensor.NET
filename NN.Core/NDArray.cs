using NN.Core.Numpy;
using NN.Native;
using NN.Native.Basic;
using NN.Native.Basic.DType;
using NN.Native.Data;
using System.Numerics;
using System.Security.Authentication.ExtendedProtection;

namespace NN.Core
{
 
#if NET7_0_OR_GREATER
    public partial class NDArray<T>: NDArray where T : unmanaged, INumber<T>
#else
    public partial class NDArray<T>: NDArray where T: unmanaged
#endif
    {
        internal NativeArray<T> _nativeArray;
        protected bool _isValid;
#if NET7_0_OR_GREATER
        public NDArray(NativeArray<T> nativeArray)
        {
            _nativeArray = nativeArray;
            _isValid = true;
        }
        public NDArray()
        {
            _isValid = false;
        }
#else
        protected NDArray(NativeArray<T> nativeArray)
        {
            _nativeArray = nativeArray;
            _isValid = true;
        }
#endif

        public override ReadOnlySpan<int> Shape { get => _nativeArray._layout.Shape.AsSpan(0, _nativeArray._layout.Ndim); }
        public override int Ndim { get => _nativeArray._layout.Ndim; }
        public override Type DType { get => typeof(T); }
        public bool IsValid { get => _isValid; }

        public override object GetAt(params int[] indices)
        {
            return _nativeArray[indices];
        }
        public T this[params int[] indices]
        {
            get => _nativeArray[indices];
            set => _nativeArray[indices] = value;
        }

        internal static NDArray<T> FromNativeArray(NativeArray<T> nativeArray)
        {
#if NET7_0_OR_GREATER
            return new NDArray<T>(nativeArray);
#else
            if (typeof(T) == typeof(int))
            {
                return (NDArray<T>)(object)(new NDArrayWithHandler<int, Int32Handler>((NativeArray<int>)nativeArray) as NDArray<int>); 
            }
            else if (typeof(T) == typeof(float))
            {
                return (NDArray<T>)(object)(new NDArrayWithHandler<float, Float32Handler>((NativeArray<float>)nativeArray) as NDArray<float>);
            }
            else if (typeof(T) == typeof(double))
            {
                return (NDArray<T>)(object)(new NDArrayWithHandler<double, Float64Handler>((NativeArray<double>)nativeArray) as NDArray<double>);
            }
            else
            {
                throw new NotImplementedException();
            }
#endif
        }
    }
}