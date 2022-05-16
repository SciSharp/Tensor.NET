using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Common;

namespace Tensornet{
    public partial class Tensor<T> where T : struct, IEquatable<T>, IConvertible{
        public Tensor<T> Where(Func<T, bool> condition, T trueCase, T falseCase){
            return OnElemOperation.Execute<T, T>(this, x =>
            {
                if (condition(x)) return trueCase;
                else return falseCase;
            });
        }
    }

    public static partial class Tensor{
        public static Tensor<T> Where<T>(Tensor<bool> conditionTensor, T trueCase, T falseCase) where T : struct, IEquatable<T>, IConvertible{
            return OnElemOperation.Execute<bool, T>(conditionTensor, x =>
            {
                if (x) return trueCase;
                else return falseCase;
            });
        }
    }
}