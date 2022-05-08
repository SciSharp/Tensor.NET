using Numnet.Exceptions;

namespace Numnet.Native.Param{
    internal struct MatmulParam{
        
    }
    internal struct DotParam{
        
    }
    internal struct PermuteParam{
        internal IntPtr dims;
    }
    internal struct TransposeParam{
        internal int dimA;
        internal int dimB;
    }
    internal struct TypeConvertParam{
        internal DType targetType;
    }
    internal struct NormalParam{
        internal double mean;
        internal double std;
    }
    internal struct UniformParam{
        internal double minValue;
        internal double maxValue;
    }
    internal struct EyeParam{
        internal int k;
    }
    internal struct FillParam{
        internal double value;
    }
    internal struct LinspaceParam{
        internal double start;
        internal double stop;
        internal int num;
        internal bool isEndpoint;
    }
    internal struct ConcatParam{
        internal int axis;
    }
}