using System.Text;
using System.Collections.Generic;

namespace CodeGenerator{
    public class BasicArithGenerator{
        public string Name{ get; set; }
        public string Expr{ get; set; }
        private List<string> _types= new List<string>(new string[]{ "double", "float", "long", "int", "bool" });
        private string[] _priority = new string[]{ "double", "float", "long", "int", "bool" };
        public string GetCode(){
            StringBuilder builder = new StringBuilder();
            foreach(var typeA in _types){
                foreach(var typeB in _types){
                    string typeRes = null;
                    for (int i = 0; i < _priority.Length; i++){
                        if(typeA == _priority[i] || typeB == _priority[i]){
                            typeRes = _priority[i];
                            break;
                        }
                    }
                    if(typeRes is null){
                        throw new System.Exception();
                    }
                    builder.Append(@"
public static Tensor<" + typeRes + $"> {Name}(this Tensor<" + typeA + @"> a, Tensor<" + typeB + @"> b){
    return InterElemOperation.Execute<" + $"{typeA}, {typeB}, {typeRes}" + $">(a, b, (x, y) => {Expr});" + @"
}");
                }
            }
            return builder.ToString();
        }
    }
}