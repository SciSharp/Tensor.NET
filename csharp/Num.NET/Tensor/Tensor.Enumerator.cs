using System.Collections;

namespace Numnet{
    public class TensorEnumerator<T> : IEnumerator, IEnumerator<T> where T : struct, IEquatable<T>, IConvertible{
        private readonly Tensor<T> _tensor;
        private int _index;
        private T _current;
        private int[] indices;

        internal TensorEnumerator(Tensor<T> tensor)
        {
            _tensor = tensor;
            _index = 0;
            _current = default;
            indices = new int[4] { 0, 0, 0, 0 };
        }

        public void Dispose()
        {
        }

        public bool MoveNext()
        {
            Tensor<T> localTensor = _tensor;

            if ((uint)_index < (uint)localTensor.TLayout.TotalElemCount())
            {
                _current = localTensor.AsSpan()[_index];
                // _index++;
                IndexIncrease();
                return true;
            }
            return MoveNextRare();
        }

        private bool MoveNextRare()
        {
            _index = _tensor.TLayout.TotalElemCount() + 1;
            _current = default;
            return false;
        }

        private void IndexIncrease(){
            indices[_tensor.TLayout.NDim - 1]++;
            _index += _tensor.TLayout.Stride[_tensor.TLayout.NDim - 1];
            for (int i = _tensor.TLayout.NDim - 1; i >= 1; i--){
                if(indices[i] < _tensor.TLayout.Shape[i]) break;
                else{
                    indices[i - 1]++;
                    _index = _index + _tensor.TLayout.Stride[i - 1] - _tensor.TLayout.Stride[i] * indices[i];
                    indices[i] = 0;
                }
            }
        }

        public T Current => _current;

        object IEnumerator.Current
        {
            get
            {
                if (_index == 0 || _index == _tensor.TLayout.TotalElemCount() + 1)
                {
                    throw new InvalidOperationException();
                }
                return Current;
            }
        }

        void IEnumerator.Reset()
        {
            _index = 0;
            _current = default;
        }
    }
}