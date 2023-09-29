#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include </usr/local/include/opencv4/opencv2/core.hpp>
#include </usr/local/include/opencv4/opencv2/highgui.hpp>


#include <iostream>
#include <stdio.h> 
#include </usr/include/boost/filesystem.hpp>
#include </usr/include/boost/range/irange.hpp>
#include </usr/include/boost/range/combine.hpp>
#include </usr/include/range/v3/all.hpp>
#include <filesystem>


#include <torch/torch.h>
#include <torch/script.h>


//using namespace cv;
//using namespace std;

/*
int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  //std::cout << torch::cuda::is_available() << std::endl;
  torch::Device device = torch::kCPU;
  std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
  if (torch::cuda::is_available()) {
	  std::cout << "CUDA is available! Training on GPU." << std::endl;
	  device = torch::kCUDA;}
}
*/

std::string root_folder_path = "/home/johnathon/Desktop/test_dir/";
std::string raw_image_folder_path = root_folder_path + "raw_images";
std::string masked_images_folder_path = root_folder_path + "masked_images";

std::vector<std::string> raw_image_path_list;
std::vector<std::string> masked_images_path_list;

// sky, land, sea, ship, buoy, other
std::vector<std::string> CLASS_LIST = {"background","sky", "land", "sea", "ship", "buoy", "other"};
//std::vector<int> CLASS_LIST_LABEL = {};

// creating a vector of object type vector with obejct type integer
std::vector<std::vector<int>> PALETTE = {{0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0}, 
                                         {0, 0, 128}, {128, 0, 128}, {0, 128, 128}};

// ---------------- function to get the list of image path ----------------
std::vector<std::string> get_list_of_file_paths(std::string folder_path);



// ---------------- function to slice the a list ----------------
std::vector<std::string> slicing_vector(std::vector<std::string> file_list,int X, int Y);



// ---------------- function to get class list label ----------------
std::vector<int> get_class_list_label(std::vector<std::string> class_list);



// ------------------------------ function to get class_indices_mask_list and one_hot_encoding_mask_list -----------------------------
torch::Tensor rgb_to_encoded_mask(std::string mask_image_path, std::vector<std::string> class_list, std::vector<std::vector<int>> colourmap);



// ---------------- function to convert opencv images into tensor images -----------------------
std::vector<torch::Tensor> convert_image_to_tensors(std::vector<std::string> image_path_list);



// ---------------- function to convert mask into tensor mask with class indices -----------------------
std::vector<torch::Tensor> convert_mask_to_tensors(std::vector<std::string> mask_path_list, std::vector<std::string> class_list, std::vector<std::vector<int>> colourmap);


class ImageDataset: public torch::data::Dataset<ImageDataset>
{
  private:
  // Declare 2 vectors of tensors for images and masks
  // initialise the images and masks
  std::vector<torch::Tensor> tensor_image_list, tensor_class_indices_mask_list;

  public:
  // Constructor
  explicit ImageDataset(std::vector<std::string> image_path_list, std::vector<std::string> mask_path_list) 
  {
    // takes from the private class
    tensor_image_list = convert_image_to_tensors(image_path_list);
    //std::cout << tensor_image_list.size() << std::endl;
    //std::cout << tensor_image_list << std::endl;

    // takes from the private class 
    tensor_class_indices_mask_list = convert_mask_to_tensors(mask_path_list, CLASS_LIST, PALETTE);
    //std::cout << tensor_class_indices_mask_list.size() << std::endl;
  };

  // Override get() function to return tensor at location index
  torch::data::Example<> get(size_t index) override
  {
    torch::Tensor tensor_image = tensor_image_list[index];
    torch::Tensor tensor_mask = tensor_class_indices_mask_list[index];
    
    return {tensor_image, tensor_mask};
  };


  // Return the length of data
  torch::optional<size_t> size() const override
  {
    return tensor_image_list.size();
  };

};
  
size_t batch_size = 1;

int main() {

  // use CPU
  torch::Device device = torch::kCPU;
  
  // if gpu available, use GPU
  if (torch::cuda::is_available()) 
  {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  }

  // get list of file path for raw images
  raw_image_path_list = get_list_of_file_paths(raw_image_folder_path);
  //std::cout << raw_image_path_list << std::endl;

  // get list of file path for masked images
  masked_images_path_list = get_list_of_file_paths(masked_images_folder_path);
  //std::cout << masked_images_path_list << std::endl;
  

  // raw images path
  // get length of raw image list
  int size_of_raw_image_list = raw_image_path_list.size();
  std::vector<std::string> train_raw_image_path_list = slicing_vector(raw_image_path_list, 0, 29);
  std::vector<std::string> val_raw_image_path_list = slicing_vector(raw_image_path_list, 30, 39);
  // std::vector<std::string> test_raw_image_path_list = slicing_vector(raw_image_path_list, 40, size_of_raw_image_list - 1);
  std::vector<std::string> test_raw_image_path_list = slicing_vector(raw_image_path_list, 49, size_of_raw_image_list - 1);
  //std::cout << test_raw_image_path_list << std::endl;

  /* 
  std::cout << train_raw_image_path_list << std::endl;
  std::cout << " " << std::endl;
  std::cout << val_raw_image_path_list << std::endl;
  std::cout << " " << std::endl;
  std::cout << test_raw_image_path_list << std::endl;
  */

  // mask images path
  // get length of mask image list
  int size_of_mask_image_list = masked_images_path_list.size();
  std::vector<std::string> train_masked_image_path_list = slicing_vector(masked_images_path_list, 0, 29);
  std::vector<std::string> val_masked_image_path_list = slicing_vector(masked_images_path_list, 30, 39);
  //std::vector<std::string> test_masked_image_path_list = slicing_vector(masked_images_path_list, 40, size_of_mask_image_list - 1);
  std::vector<std::string> test_masked_image_path_list = slicing_vector(masked_images_path_list, 49, size_of_mask_image_list - 1);
  //std::cout << test_masked_image_path_list << std::endl;
  


  std::vector<int> CLASS_LIST_LABEL = get_class_list_label(CLASS_LIST);
  //std::cout << CLASS_LIST_LABEL << std::endl;
  
  /* 
  // testing function
  std::string mask_image_path = "/home/johnathon/Desktop/test_dir/masked_images/0_mask.png";
  torch::Tensor class_indices_mask_list = rgb_to_encoded_mask(mask_image_path, CLASS_LIST, PALETTE);
  std::cout << (class_indices_mask_list)<< std::endl;
  */

  // Generate dataset. At this point you can add transforms to your dataset, e.g. stack batches into a single tensor.
  auto test_dataset = ImageDataset(test_raw_image_path_list, test_masked_image_path_list).map(torch::data::transforms::Stack<>());
  //auto test_dataset = ImageDataset(test_raw_image_path_list, test_masked_image_path_list);

  //std::cout << test_dataset.size() << std::endl;
  
  // Generate a data loader.
  auto test_dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), batch_size);
  //std::cout << test_dataloader << std::endl;


  // In a for loop you can now use your data.
  for (auto& batch : *test_dataloader) 
  {
    auto data = batch.data;
    auto target = batch.target;
    //std::cout << data.sizes() << std::endl;

    // convert data to float and put it to device
    data = data.to(torch::kF32).to(device);
    target = target.to(torch::kF32).to(device);
    //std::cout << data << std::endl;

    // load model
    torch::jit::script::Module module;
    module = torch::jit::load("/home/johnathon/Desktop/multi_segmentation/best_model_weights/epoch_22_serial_module_lowest_validation_loss_model.pt");
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(data);
    at::Tensor output = module.forward(inputs).toTensor();
    //std::cout << output << "\n";
    //std::cout << output.sizes() << "\n";

    torch::Tensor predicted_class_indices = output.argmax(1);
    //std::cout << predicted_class_indices << std::endl;

    std::map <int, std::vector<int>> mapping = {};

    for (int i=0; i<CLASS_LIST.size(); ++i)
    {
      //std::cout << PALETTE.at(i) << i << "\n";

      // PALETTE.at(i) --> get value at index i
      mapping.insert({i, PALETTE.at(i)});
    }

    torch::Tensor new_predictions = predicted_class_indices.permute({1, 2, 0}).contiguous();
    //std::cout << new_predictions.sizes() << std::endl;

    torch::Tensor three_channels_prediction = torch::cat({new_predictions, new_predictions, new_predictions},2);
    //std::cout << three_channels_prediction << std::endl;
    //std::cout << three_channels_prediction.sizes() << std::endl;

    torch::Tensor new_mask = torch::empty(three_channels_prediction.sizes());
    //std::cout << new_mask << std::endl;

    for (auto const& i : mapping) 
    {
      std::vector<int> colour_palette = i.second;
      int length_of_colour_palette = colour_palette.size();

      // tensor option object type to integer
      auto options = torch::TensorOptions().dtype(torch::kInt32);
      
      // create a tensor type colour palette
      torch::Tensor tensor_colour_palette = torch::from_blob(colour_palette.data(), {length_of_colour_palette}, options).clone();
      //std::cout << tensor_colour_palette << std::endl;

      // if not class indices, remain, else, change class indices to colour code
      three_channels_prediction = torch::where(three_channels_prediction != i.first, three_channels_prediction.to(device), tensor_colour_palette.to(device));
      //break;
    }

    // std::cout << new_mask << std::endl;
    std::cout << three_channels_prediction.sizes() << std::endl;

    three_channels_prediction = three_channels_prediction.to(torch::kU8);
    //std::cout << new_mask << std::endl;

    three_channels_prediction = three_channels_prediction.to(torch::kCPU);

    int height = three_channels_prediction.sizes()[0];
    int width = three_channels_prediction.sizes()[1];

    three_channels_prediction = three_channels_prediction.reshape({width * height * 3});
    //std::cout << new_mask << std::endl;

    cv::Mat predicted_mask_output = cv::Mat(cv::Size(width, height), CV_8UC3, three_channels_prediction.data_ptr());
    //std::cout << mat << std::endl;
    //std::cout << mat.size() << std::endl;

    cv::imwrite("/home/johnathon/Desktop/test_cpp.jpg", predicted_mask_output);
    cv::waitKey(0); 
    

    /* new_mask = new_mask.to(torch::kCPU).to(torch::kInt32);
    //new_mask = new_mask.reshape({960 * 1280 * 3});
    cv::Mat mat = cv::Mat(960, 1280, CV_8UC3, new_mask.data_ptr<int32_t>()).clone();

    std::cout << mat << std::endl;
    std::cout << mat.size() << std::endl;

    cv::imwrite("/home/johnathon/Desktop/test_cpp.jpg" ,mat);
    cv::waitKey(0); */
    // break;
   } 

  /* 
  //declaring a matrix named BGR_image
  cv::Mat BGR_image, RGB_image;
  //loading the image in the matrix
  BGR_image = imread("/home/johnathon/Desktop/anotated_frames/1660.png", cv::IMREAD_COLOR);
  // convert BGR to RGB
  cv::cvtColor(BGR_image, RGB_image, cv::COLOR_BGR2RGB);

  std::cout << RGB_image << std::endl;
 
  std::cout << "Image Width : " << RGB_image.size[1] << std::endl;
  std::cout << "Image Height: " << RGB_image.size[0] << std::endl;
  std::cout << "Image Size: " << RGB_image.size << std::endl;
  std::cout << "Image Channels: " << RGB_image.channels() << std::endl;
  
  // check the pixel value at position y=0, x=0
  std::cout << "Image: " << RGB_image.at<cv::Vec3b>(0,0) << std::endl;

  // get the each colour value of an individual pixel
  //Vec3b intensity = RGB_image.at<Vec3b>(0, 0);
  //float blue = intensity.val[0];
  //float green = intensity.val[1];
  //float red = intensity.val[2];
  */
}

//------------------------------------------- Fuctions -----------------------------------------------------------------------

//------------------------------ get a list of file paths ------------------------------
std::vector<std::string> get_list_of_file_paths(std::string folder_path)
{ 
  // create an empty vector with object type string in it
  std::vector<std::string> list_of_file_path;
  
  // file system
  std::vector<std::filesystem::path> files_in_directory;
  std::copy(std::filesystem::directory_iterator(folder_path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
  // sort the files
  std::sort(files_in_directory.begin(), files_in_directory.end());
  for (const std::string& file_path : files_in_directory) {
      // std::cout << filename << std::endl; // printed in alphabetical order
      list_of_file_path.push_back(file_path);
  }

  return list_of_file_path;
}


// ------------------------------ slicing of a vector list ------------------------------
std::vector<std::string> slicing_vector(std::vector<std::string> file_list,int X, int Y)
{
 
    // Starting and Ending iterators
    auto start = file_list.begin() + X;
    auto end = file_list.begin() + Y + 1;
 
    // To store the sliced vector
    std::vector<std::string> result(Y - X + 1);
 
    // Copy vector using copy function()
    copy(start, end, result.begin());
 
    // Return the final sliced vector
    return result;
}

int function_name(int a, int b) {
  return a +  b;
}


// ------------------------------ get class list label ------------------------------
std::vector<int> get_class_list_label(std::vector<std::string> class_list)
{
  int length_of_class_list = class_list.size();

  // initialise an empty vector
  std::vector<int> empty_list;

  for (auto i : boost::irange(0, length_of_class_list))
  {
    //std::cout << i << "\n";
    empty_list.push_back(i);
  }

  return empty_list;
}


// --------------------------------- get class_indices_mask_list and one_hot_encoding_mask_list -------------------------------------------
torch::Tensor rgb_to_encoded_mask(std::string mask_image_path, std::vector<std::string> class_list, std::vector<std::vector<int>> colourmap)
{
  cv::Mat BGR_image, RGB_mask;
  BGR_image = imread(mask_image_path, cv::IMREAD_COLOR);
  
  // convert BGR to RGB
  cv::cvtColor(BGR_image, RGB_mask, cv::COLOR_BGR2RGB);


  // tensor mask shape --> [960, 1280,3]
  torch::Tensor tensor_mask = torch::from_blob(RGB_mask.data, { RGB_mask.rows, RGB_mask.cols, 3 }, torch::kUInt8);
  //std::cout << tensor_mask.size() << std::endl;


  // tensor mask shape --> [3, 960, 1280]
  tensor_mask = tensor_mask.permute({2, 0, 1}).contiguous();
  // get size of tensor
  //std::cout << tensor_mask.sizes() << std::endl;

  // create a mapping of colour to class indices
  // class indices is the key, colour code is the value --> e.g (2, 0, 128, 0)
  std::map <int, std::vector<int>> mapping = {};

  //std::cout << PALETTE.at(0) << std::endl;
  
  for (int i=0; i<class_list.size(); ++i)
  {
    //std::cout << PALETTE.at(i) << i << "\n";

    // PALETTE.at(i) --> get value at index i
    mapping.insert({i, colourmap.at(i)});
  }

  //std::cout << mapping << std::endl;

  // empty tensor to concatenate tensor masks of objects with class indices encoding
  // create an empty tensor of zeros of size (1,H,W) --> 1 is class indice 0, which is the background class --> (0, 0,0,0)
  // RGB_mask.size[0] --> Height (960)
  // RGB_mask.size[1] --> Width (1280)
  torch::Tensor class_indices_mask_list = torch::zeros({1, RGB_mask.size[0], RGB_mask.size[1]});

  // empty tensor to concatenate tensor masks of objects with one hot encoding
  // create an empty tensor of zeros of size (1,H,W) --> 1 is class indice 0, which is the background class --> (0, 0,0,0)
  // RGB_mask.size[0] --> Height (960)
  // RGB_mask.size[1] --> Width (1280)
  torch::Tensor one_hot_encoding_mask_list = torch::zeros({1, RGB_mask.size[0], RGB_mask.size[1]});

  for (auto const& i : mapping) 
  {
    //std::cout << i << std::endl;

    // if the class indice == 0, pass
    if (i.first == 0)
    {
      continue;
    }
    
    // do calculation for other class indices
    else 
    {
      // i.first --> key for mapping dictionary
      // i.second --> value for mapping dictionary
      //std::cout << i.first << std::endl;
      // std::cout << i.second << std::endl;
      torch::Tensor tensor_array = torch::tensor(i.second);
      //std::cout << tensor_array.unsqueeze(1).unsqueeze(2).sizes() << std::endl;

      // e.g. class label 3
      // e.g. tensor_array --> [3]
      // e.g. tensor_array.unsqueeze(1) --> [3,1]
      // e.g. tensor_array.unsqueeze(1).unsqueeze(2) --> [3,1,1]
      torch::Tensor idx = (tensor_mask == tensor_array.unsqueeze(1).unsqueeze(2));
      //std::cout << idx << std::endl;
      //std::cout << idx.sizes() << std::endl;

      // Check that all channels match
      // shape --> [960, 1280]
      torch::Tensor validx = (idx.sum(0) == 3);
      // std::cout << validx << std::endl;

      //convert the validx to torch integer and then concatenate to the empty tensor
      // .unsqueeze(0) --> a increase the dimension of the tensor by 1 at position 0
      // 0 --> concatenate at position 1
      one_hot_encoding_mask_list = torch::cat({one_hot_encoding_mask_list, validx.to(torch::kInt64).unsqueeze(0)},0);
      
      // transform the class indice to tensor
      torch::Tensor class_indice = torch::tensor(i.first);
      //std::cout << tensor << std::endl;

      // where the value is 1 (true), replace it with class indice and the rest is kept as 0
      torch::Tensor class_indice_mask = torch::where(validx == 1, class_indice, 0);
      
      //convert the validx to torch integer and then concatenate to the empty tensor
      // .unsqueeze(0) --> a increase the dimension of the tensor by 1 at position 0
      // 0 --> concatenate at position 1
      class_indices_mask_list = torch::cat({class_indices_mask_list, class_indice_mask.unsqueeze(0)},0);
    
      //break;
    
    }
    
  }
  // can also return one_hot_encoding_mask_list for one hot encoding format
  return class_indices_mask_list;
}


// ---------------- convert opencv images to tensor images -----------------------
std::vector<torch::Tensor> convert_image_to_tensors(std::vector<std::string> image_path_list)
{
  std::vector<torch::Tensor> tensor_img_list = {};

  for (int i=0; i<image_path_list.size(); ++i)
  {
    //std::cout << image_path_list[i] << std::endl;
    
    //declaring a matrix named BGR_image
    cv::Mat BGR_image, RGB_image;
    
    //loading the image in the matrix
    BGR_image = imread(image_path_list[i], cv::IMREAD_COLOR);

    
    
    // convert BGR to RGB
    cv::cvtColor(BGR_image, RGB_image, cv::COLOR_BGR2RGB);
    //std::cout <<  RGB_image << std::endl;
    
    // tensor mask shape --> [960, 1280,3]
    torch::Tensor tensor_mask = torch::from_blob(RGB_image.data, { RGB_image.rows, RGB_image.cols, 3 }, torch::kUInt8);
    //std::cout <<  tensor_mask << std::endl;

    // tensor mask shape --> [3, 960, 1280]
    tensor_mask = tensor_mask.permute({2, 0, 1}).contiguous();
    //tensor_mask = torch::data::transforms::TensorTransform(tensor_mask)
    //std::cout <<  tensor_mask << std::endl;

    //std::cout <<  tensor_mask.div(255) << std::endl;

    // .div(255) --> same as transform.toTensor()
    tensor_img_list.push_back(tensor_mask.div(255));

    //break;
  }

  return tensor_img_list;
}


// ---------------- convert mask to tensor mask with class indices -----------------------
std::vector<torch::Tensor> convert_mask_to_tensors(std::vector<std::string> mask_path_list, std::vector<std::string> class_list, std::vector<std::vector<int>> colourmap)
{
  std::vector<torch::Tensor> tensor_mask_list = {};

  for (int i=0; i<mask_path_list.size(); ++i)
  {
    //std::cout << mask_path_list[i] << std::endl;

    torch::Tensor tensor_mask = rgb_to_encoded_mask(mask_path_list[i], class_list, colourmap);

    //std::cout << tensor_mask << std::endl;
    //std::cout << tensor_mask.sizes() << std::endl;

    tensor_mask_list.push_back(tensor_mask);
    //tensor_mask_list.insert(tensor_mask_list.end(), tensor_mask);

    //break;
  }

  return tensor_mask_list;
}