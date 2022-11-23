using NN.Native.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Core
{
    public partial class NDArray<T>
    {
        public static class Random
        {
            public static NDArray<T> Normal(int[] shape, T minValue, T maxValue)
            {
                var nativeArray = NativeArray.Random.Normal<T>(new Native.Basic.NativeLayout(shape), minValue, maxValue);
                return new NDArray<T>(nativeArray);
            }
        }
    }
}
