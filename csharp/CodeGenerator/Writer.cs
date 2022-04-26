using System.IO;

namespace CodeGenerator{
    public class CodeWriter{
        public static string Prefix { get; set; } = "Results";
        public static void Write(string Name, string code){
            Name += ".csg";
            Name = Prefix + "/" + Name;
            FileStream fs = new FileStream(Name, FileMode.OpenOrCreate, FileAccess.ReadWrite);
            StreamWriter sw = new StreamWriter(fs);
            sw.Write(code);
            sw.Close();
            fs.Close();
        }
    }
}