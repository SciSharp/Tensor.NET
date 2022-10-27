using NN.Native.Abstraction.Operators;
using NN.Native.Basic;
using NN.Native.Operators.Common.Params;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Operators.Common
{
    public abstract class TypeConvertOperatorBase
    {
        public static void DeduceLayout(in NativeLayout src, in TypeConvertParam param, out NativeLayout dst)
        {
            dst = new(src._shape);
        }
    }
}
