using NN.Native.Abstraction.DType;
using NN.Native.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace NN.Core
{
    //#if !NET7_0_OR_GREATER
    //    internal class NDArrayWithHandler<TData, THandler>: NDArray<TData> where TData : unmanaged where THandler : INativeDTypeHandler<TData>
    //    {
    //        public NDArrayWithHandler(NativeArray<TData> nativeArray): base(nativeArray)
    //        {

//        }
//    }
//#endif
#if !NET7_0_OR_GREATER
    internal class NDArrayWithHandler<TData, THandler> : NDArray<TData> where TData : unmanaged where THandler : INativeDTypeHandler<TData>
    {
        public NDArrayWithHandler(NativeArray<TData> nativeArray) : base(nativeArray)
        {

        }
    }
#endif

}
