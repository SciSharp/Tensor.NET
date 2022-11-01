using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.BenchmarkTest.Helper
{
    public class DimensionHelper
    {
        private int[][] _dimensions;
        public DimensionHelper(params int[][] dimensions)
        {
            _dimensions = dimensions;
        }
        public int[] this[int idx]
        {
            get => _dimensions[idx];
        }
        public override string ToString()
        {
            string res = "";
            foreach (var dim in _dimensions)
            {
                res += $"({string.Join(',', dim)}) ";
            }
            return res;
        }
    }
}
