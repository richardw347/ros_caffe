/*
 * ros_caffe_test.cpp
 *
 *  Created on: Aug 31, 2015
 *      Author: Tzutalin
 */

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <spark_msgs/CaffeClassify.h>
#include "Classifier.h"

const std::string RECEIVE_IMG_TOPIC_NAME = "camera/rgb/image_raw";
const std::string PUBLISH_RET_TOPIC_NAME = "/caffe_ret";

Classifier* classifier;
std::string model_path;
std::string weights_path;
std::string mean_file;
std::string label_file;
std::string image_path;

//ros::Publisher gPublisher;
ros::ServiceServer classifyServiceServer;

/*
void publishRet(const std::vector<Prediction>& predictions);

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        //cv::imwrite("rgb.png", cv_ptr->image);
		cv::Mat img = cv_ptr->image;
		std::vector<Prediction> predictions = classifier->Classify(img);
		publishRet(predictions);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}*/


/*void publishRet(const std::vector<Prediction>& predictions)  {
    std_msgs::String msg;
    std::stringstream ss;
    for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        ss << "[" << p.second << " - " << p.first << "]" << std::endl;
    }
    msg.data = ss.str();
    gPublisher.publish(msg);
}*/


bool classifyServCallback(spark_msgs::CaffeClassify::Request& request, spark_msgs::CaffeClassify::Response& response){
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(request.input_image, "bgr8");
        cv::Mat img = cv_ptr->image;
        std::vector<Prediction> predictions = classifier->Classify(img);
        size_t pred_size = predictions.size();
        for (size_t i=0; i<pred_size; i++){
            response.labels[i] = predictions[i].first;
            response.probabilities[i] = predictions[i].second;
        }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", request.input_image.encoding.c_str());
    }
    return true;
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "ros_caffe_test");
    ros::NodeHandle nh;
    // image_transport::ImageTransport it(nh);
    // To receive an image from the topic, PUBLISH_RET_TOPIC_NAME
    // image_transport::Subscriber sub = it.subscribe(RECEIVE_IMG_TOPIC_NAME, 1, imageCallback);
	// gPublisher = nh.advertise<std_msgs::String>(PUBLISH_RET_TOPIC_NAME, 100);
    const std::string ROOT_SAMPLE = ros::package::getPath("ros_caffe");
    model_path = ROOT_SAMPLE + "/data/deploy.prototxt";
    weights_path = ROOT_SAMPLE + "/data/bvlc_reference_caffenet.caffemodel";
    mean_file = ROOT_SAMPLE + "/data/imagenet_mean.binaryproto";
    label_file = ROOT_SAMPLE + "/data/synset_words.txt";
    image_path = ROOT_SAMPLE + "/data/cat.jpg";

    classifier = new Classifier(model_path, weights_path, mean_file, label_file);

    // Test data/cat.jpg
    cv::Mat img = cv::imread(image_path, -1);
    std::vector<Prediction> predictions = classifier->Classify(img);

    /* Print the top N predictions. */
    std::cout << "Test default image under /data/cat.jpg" << std::endl;
    for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
    }

    ROS_INFO("Classification service started...");
    classifyServiceServer = nh.advertiseService("spark_recognition/caffe_classify", classifyServCallback);
    ros::spin();
    delete classifier;
    ros::shutdown();
    return 0;
}
