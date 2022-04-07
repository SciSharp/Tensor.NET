using Numnet.Native;

namespace Numnet.Common{
    internal interface ITypeDeduce{
        Type Deduce(DType lhs, DType rhs);
    }

    internal class DefaultTypeDeduce:ITypeDeduce{
        public Type Deduce(DType lhs, DType rhs){
            if(lhs == rhs) return TensorTypeInfo.GetTypeInfo(lhs);
            if(lhs == DType.Float64 || rhs == DType.Float64){
                return typeof(Double);
            }
            else if(lhs == DType.Float32 || rhs == DType.Float32){
                return typeof(float);
            }
            else if(lhs == DType.Int64 || rhs == DType.Int64){
                return typeof(Int64);
            }
            else if(lhs == DType.Int32 || rhs == DType.Int32){
                return typeof(Int32);
            }
            else if(lhs == DType.Bool || rhs == DType.Bool){
                return typeof(Boolean);
            }
            else{
                throw new NotImplementedException("The input types are not supported so far. Please choose among bool, int, long, float and double.");
            }
        }
    }
}