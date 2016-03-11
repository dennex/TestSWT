using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using System.IO;


namespace testSWT
{
	class Program
	{
		static void Main( string[] args )
		{
			// get all images from C:\Users\Anne\Documents\MATLAB\distTrans\raw_image
			string directory = "C:\\Users\\Denis\\Documents\\MATLAB\\distTrans\\raw_image";
			List<string> imagePaths = GetImagesPath(directory);

			//foreach( string str in imagePaths )
			for( int i = 0; i < imagePaths.Count();i++ )
			{
				Console.WriteLine( "Hello world" );
				IplImage testImg = Cv.LoadImage( imagePaths[i] );
				double ratio = testImg.Height/176.0;
				int width = (int)((double)testImg.Width /ratio);
				IplImage resizedImg = Cv.CreateImage( new CvSize( width, 176 ), BitDepth.U8, 3 );
				Cv.Resize(testImg, resizedImg,Interpolation.Cubic);
				IplImage output = SWT.textDetection( resizedImg, false );
				// put result in /result
				string direct = Path.GetDirectoryName( imagePaths[i] );
				string fileName = Path.GetFileName( imagePaths[i] );
				string path = GetFilePath( direct + "\\output\\", fileName );
				Cv.SaveImage( path, output );
			}
		}

		public static string GetFilePath(string dir, string fileName)
		{
			if( Path.GetFileName( fileName ) != fileName )
			{
				throw new Exception( "'fileName' is invalid!" );
			}
			string combined = Path.Combine( dir, fileName );
			return combined;
		}

		public static List<String> GetImagesPath( String folderName )
		{

			DirectoryInfo Folder;
			FileInfo[] Images;

			Folder = new DirectoryInfo( folderName );
			Images = Folder.GetFiles();
			List<String> imagesList = new List<String>();

			for( int i = 0; i < Images.Length; i++ )
			{
				imagesList.Add( String.Format( @"{0}/{1}", folderName, Images[ i ].Name ) );
				// Console.WriteLine(String.Format(@"{0}/{1}", folderName, Images[i].Name));
			}


			return imagesList;
		}
	}
}
