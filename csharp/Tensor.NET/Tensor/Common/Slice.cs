using Tensornet.Exceptions;

namespace Tensornet{
    public struct Slice{
        public int Begin{ get; set; }
        public int End{ get; set; }
        public int Step{ get; set; }
        public Slice(int dim){
            Begin = dim;
            End = dim;
            Step = 0;
        }
        public Slice(int begin, int end, int step = 1){
            Begin = begin;
            End = end;
            Step = step;
        }
        public Slice(Range range){
            Begin = range.Start.IsFromEnd ? -range.Start.Value : range.Start.Value;
            End = range.End.IsFromEnd ? -range.End.Value : range.End.Value;
            Step = 1;
        }
        public Slice(Index index){
            Begin = index.IsFromEnd ? -index.Value : index.Value;
            End = Begin;
            Step = 0;
        }
        public static implicit operator Slice(Range range){
            return new Slice(range);
        }
        public static implicit operator Slice(Index index){
            return new Slice(index);
        }
        public static implicit operator Slice(int index){
            return new Slice(index);
        }
    }

    public sealed partial class TensorLayout{
        internal TensorLayout ApplySlice(Slice s, int axis){
            TensorLayout res = new TensorLayout(this);
            if(axis < 0) axis += res.NDim;
            if(axis < 0){
                throw new InvalidArgumentException($"Invalid axis to slice.");
            }
            if(axis >= res.NDim){
                throw new InvalidArgumentException($"Axis to slice exceeds the limit. The axis is {axis}, the limit is {res.NDim - 1}.");
            }
            int axisDim = res.Shape[axis];
            int begin = s.Begin;
            int end = s.End;
            int step = s.Step;
            if(step < 0){
                throw new InvalidArgumentException($"The step of slice cannot be negative.");
            }
            
            if(begin < 0) begin += axisDim;
            if(end < 0) end += axisDim;
            if(end < begin){
                throw new InvalidArgumentException($"The end index of slice should be larger than the begin index. But the begin is {begin}, the end is {end}.");
            }
            // is not scalar
            if(step > 0){
                res.Shape[axis] = (end - begin + step - 1) / step;
                res.Offset += Stride[axis] * begin;
                res.Stride[axis] *= step;
            }
            // If it's a scalar, the begin must be equal to end, and the step = 0.
            else if(begin == end){
                res.Shape[axis] = 1;
                res.Offset += Stride[axis] * begin;
            }
            else{
                throw new InvalidArgumentException($"Bad slice. It may be caused by setting step = 0 but begin != end.");
            }

            return res;
        }
    }

    public partial class Tensor<T>{
        public Tensor<T> this[params Slice[] slices]{
            get{
                if(slices.Length > TLayout.NDim){
                    throw new InvalidArgumentException("Too many dims of slice.");
                }
                else if(slices.Length < TLayout.NDim){
                    throw new InvalidArgumentException("Too few dims of slice.");
                }
                TensorLayout layout = new TensorLayout(TLayout);
                for (int i = 0; i < slices.Length; i++){
                    layout = layout.ApplySlice(slices[i], i);
                }
                for (int i = slices.Length - 1; i >= 0; i--){
                    if(slices[i].Step == 0 && slices[i].Begin == slices[i].End){
                        layout.RemoveAxisInplace(i);
                    }
                }
                return new Tensor<T>(TMemory, layout);
            }
        }
        public Tensor<T> this[params Range[] slices]{
            get{
                if(slices.Length > TLayout.NDim){
                    throw new InvalidArgumentException("Too many dims of slice.");
                }
                else if(slices.Length < TLayout.NDim){
                    throw new InvalidArgumentException("Too few dims of slice.");
                }
                TensorLayout layout = new TensorLayout(TLayout);
                for (int i = 0; i < slices.Length; i++){
                    layout = layout.ApplySlice(new Slice(slices[i]), i);
                }
                for (int i = slices.Length - 1; i >= 0; i--){
                    if(slices[i].Start.Value == slices[i].End.Value && slices[i].Start.IsFromEnd == slices[i].End.IsFromEnd){
                        layout.RemoveAxisInplace(i);
                    }
                }
                return new Tensor<T>(TMemory, layout);
            }
        }
    }
}