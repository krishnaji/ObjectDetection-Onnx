using Microsoft.ML.Transforms.Image;
using System.Drawing;

namespace OnnxObjectDetection
{
    public struct ImageSettings
    {
        public const int imageHeight = 224;
        public const int imageWidth = 224;
    }

    public class ImageInputData
    {
        [ImageType(ImageSettings.imageHeight, ImageSettings.imageWidth)]
        public Bitmap Image { get; set; }
    }
}
