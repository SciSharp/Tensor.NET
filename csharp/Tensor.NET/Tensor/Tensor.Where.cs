using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Common;

namespace Tensornet{
    public partial class Tensor<T> where T : struct, IEquatable<T>, IConvertible{
        /// <summary>
        /// Return elements chosen from trueCase or falseCase depending on condition.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.where.html?highlight=where
        /// </summary>
        /// <param name="condition"> The condition to decide whether use trueCase or falseCase</param>
        /// <param name="trueCase"> The value to fill when the condition is true. </param>
        /// <param name="falseCase"> The value to fill when the condition is false. </param>
        /// <returns></returns>
        public Tensor<T> Where(Func<T, bool> condition, T trueCase, T falseCase){
            return OnElemOperation.Execute<T, T>(this, x =>
            {
                if (condition(x)) return trueCase;
                else return falseCase;
            });
        }
    }

    public static partial class Tensor{
        /// <summary>
        /// Return elements chosen from trueCase or falseCase depending on condition.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.where.html?highlight=where
        /// </summary>
        /// <param name="condition"> The condition to decide whether use trueCase or falseCase</param>
        /// <param name="trueCase"> The value to fill when the condition is true. </param>
        /// <param name="falseCase"> The value to fill when the condition is false. </param>
        /// <returns></returns>
        public static Tensor<T> Where<T>(Tensor<bool> conditionTensor, T trueCase, T falseCase) where T : struct, IEquatable<T>, IConvertible{
            return OnElemOperation.Execute<bool, T>(conditionTensor, x =>
            {
                if (x) return trueCase;
                else return falseCase;
            });
        }
    }
}