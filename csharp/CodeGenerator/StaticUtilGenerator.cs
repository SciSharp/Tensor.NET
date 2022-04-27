using System.Text;
using System.Collections.Generic;

namespace CodeGenerator{
    public class StaticUtilGenerator{
        public string Name{ get; set; }
        public string Expr{ get; set; }
        public List<string> Parameters{ get; set; }
        private List<string> _types= new List<string>(new string[]{ "double", "float", "long", "int", "bool" });
        private string _targetType = null;
        public string GetCode(){
            StringBuilder builder = new StringBuilder();
            foreach(var typeInp in _types){
                string typeRes = _targetType??typeInp;
                builder.Append(@"
public static Tensor<" + typeRes + $"> {Name}(Tensor<" + typeInp + @"> inp" + (Parameters is null ? "": ", ") + 
                                    (Parameters is null ? "": string.Join(',', Parameters)) + @"){
    return OnElemOperation.Execute<" + $"{typeInp}, {typeRes}" + $">(inp, x => {Expr});" + @"
}");
            }
            return builder.ToString();
        }
    }

    public class ReversedStaticUtilGenerator{
        public string Name{ get; set; }
        public string Expr{ get; set; }
        public List<string> Parameters{ get; set; }
        private List<string> _types= new List<string>(new string[]{ "double", "float", "long", "int", "bool" });
        private string _targetType = null;
        public string GetCode(){
            StringBuilder builder = new StringBuilder();
            foreach(var typeInp in _types){
                string typeRes = _targetType??typeInp;
                builder.Append(@"
public static Tensor<" + typeRes + $"> {Name}({(Parameters is null ? "": string.Join(',', Parameters))}, Tensor<" + typeInp + @"> inp"
                                     + @"){
    return OnElemOperation.Execute<" + $"{typeInp}, {typeRes}" + $">(inp, x => {Expr});" + @"
}");
            }
            return builder.ToString();
        }
    }
}