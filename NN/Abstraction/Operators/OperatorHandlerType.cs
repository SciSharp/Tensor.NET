using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.Operators
{
    public enum OperatorHandlerType
    {
        Default = 0,
        Naive,
        X86,
        MultiThreads
    }
}
