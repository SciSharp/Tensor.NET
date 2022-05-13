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
    internal struct ArgmxxParam{
        internal int axis;
        internal bool isMax;
    }
    internal struct RepeatParam{
        internal int repeats;
        internal int axis;
    }
    internal struct FlipParam{
        internal IntPtr dims;
    }
    internal struct RotateParam{
        internal int k;
        internal int dimA;
        internal int dimB;
        internal RotateParam(int k, int dimA, int dimB){
            this.k = k;
            if(dimA > dimB){
                (this.dimA, this.dimB) = (dimB, dimA);
            }
            else{
                (this.dimA, this.dimB) = (dimA, dimB);
            }
        }
    }
    internal struct PadParam{
        internal PadMode mode;
        internal int size;
        internal IntPtr widths;
        internal IntPtr constants;
    }
    internal struct SortParam{
        internal int axis;
        internal SortOrder order;
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