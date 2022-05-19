using System.IO;
using System.Text;
using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Exceptions;

namespace Tensornet{
    public static class TensorReader{
        private static byte[] _magicNumber;
        private static byte[] _versionNumber;

        static TensorReader(){
            _magicNumber = Encoding.ASCII.GetBytes("Tensor.NET");
            _versionNumber = Encoding.ASCII.GetBytes(Tensor.VersionNumber);
        }

        public static Tensor<T> Read<T>(string path) where T : struct, IEquatable<T>, IConvertible{
            TensorLayout layout;
            var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            CheckMagicAndVersion(fs);
            layout = GetLayoutFromHeader(fs);
            var res = layout.DType switch
            {
                DType.Int32 => new Tensor<int>(ReadDataInternal<int>(fs, layout), layout).ToTensor<T>(),
                DType.Int64 => new Tensor<long>(ReadDataInternal<long>(fs, layout), layout).ToTensor<T>(),
                DType.Float32 => new Tensor<float>(ReadDataInternal<float>(fs, layout), layout).ToTensor<T>(),
                DType.Float64 => new Tensor<double>(ReadDataInternal<double>(fs, layout), layout).ToTensor<T>(),
                DType.Bool => new Tensor<bool>(ReadDataInternal<bool>(fs, layout), layout).ToTensor<T>(),
                _ => throw new NNSerializeException("Unsupported data type in serialization.")
            };
            fs.Close();
            return res;
        }

        private static void CheckMagicAndVersion(FileStream fs){
            for (int i = 0; i < _magicNumber.Length; i++){
                if(fs.ReadByte() != _magicNumber[i]){
                    throw new NNSerializeException("Wrong magic number.");
                }
            }
            fs.Position += _versionNumber.Length;
        }

        private static TensorLayout GetLayoutFromHeader(FileStream fs){
            var int32Buffer = new byte[4];
            TensorLayout res = new TensorLayout();
            fs.Read(int32Buffer, 0, 4);
            res.DType = (DType)BitConverter.ToInt32(int32Buffer, 0);
            fs.Read(int32Buffer, 0, 4);
            res.NDim = BitConverter.ToInt32(int32Buffer, 0);
            for (int i = 0; i < TensorLayout.MAX_NDIM; i++)
            {
                fs.Read(int32Buffer, 0, 4);
                res.Shape[i] = BitConverter.ToInt32(int32Buffer, 0);
            }
            res.InitContiguousLayout();
            return res;
        }

        private static void ReadContiguousData<T>(FileStream fs, TensorMemory<T> memory) where T : struct, IEquatable<T>, IConvertible{
            fs.Read(memory.AsByteSpan());
        }

        private static TensorMemory<T> ReadDataInternal<T>(FileStream fs, TensorLayout layout) where T : struct, IEquatable<T>, IConvertible{
            TensorMemory<T> res = new TensorMemory<T>(layout.TotalElemCount());
            ReadContiguousData(fs, res);
            return res;
        }

    }
}