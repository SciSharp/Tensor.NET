using Numnet.Native;

namespace Numnet.Common{
    internal static class DefaultTypeDeduce{
        public static DType Deduce(DType lhs, DType rhs){
            if(lhs == rhs) return lhs;
            if(lhs == DType.Float64 || rhs == DType.Float64){
                return DType.Float64;
            }
            else if(lhs == DType.Float32 || rhs == DType.Float32){
                return DType.Float32;
            }
            else if(lhs == DType.Int64 || rhs == DType.Int64){
                return DType.Int64;
            }
            else if(lhs == DType.Int32 || rhs == DType.Int32){
                return DType.Float32;
            }
            else if(lhs == DType.Bool || rhs == DType.Bool){
                return DType.Bool;
            }
            else{
                throw new NotImplementedException("The input types are not supported so far. Please choose among bool, int, long, float and double.");
            }
        }
    }
}