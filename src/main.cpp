#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <csignal>
#include <unistd.h>
#include <boost/program_options.hpp>
#include <H5Cpp.h>
#include <opencv2/opencv.hpp>

#include "plot.hpp"
#include "h5sx.hpp"


namespace po = boost::program_options;

#define USE_CL
#ifdef USE_CL
typedef cv::UMat MMat;
#else
typedef cv::Mat MMat;
#endif

H5File h5flow, h5contour;

po::variables_map parse(int argc, char **argv) {
	po::options_description op("Command line options");
	op.add_options()
	("version,v", "print version string")
	("help,h", "produce this help message")
	("info,p", "print information about the input file")
	("config,c", po::value< std::vector<std::string> >(), "path of a config file");

	po::positional_options_description p;
	p.add("file", -1);

	po::options_description fop;
	fop.add_options()
	("input,i", po::value<std::string>()->required(), "path of the input file")
	("background,b", po::value<std::string>(), "path of the input background image")
	("invert", po::bool_switch(), "invert background")
	("data,d", po::value<std::string>()->implicit_value(""), "path of the output hdf5 data file")
	("movie,m", po::value<std::string>()->implicit_value(""), "path of the output video")
	("live,l", po::bool_switch(), "display live window?")
	("frame.start", po::value<int>()->default_value(1), "first frame of interest")
	("frame.stop", po::value<int>(), "last frame of interest")
	("frame.step", po::value<int>()->default_value(1), "step between frames of interest")
	("frame.count", po::value<int>(), "number of frames of interest")
	("grid.width", po::value<int>()->default_value(128), "number of horizontal grid points")
	("grid.height", po::value<int>()->default_value(64), "number of vertical grid points");
	op.add(fop);

	// Parse command line
	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(op).positional(p).run(), vm);

	// Parse config files
	if (vm.count("config")) {
		std::vector<std::string> files = vm["config"].as< std::vector<std::string> >();
		for (std::vector<std::string>::const_iterator i = files.begin(); i != files.end(); ++i)
		{
			std::ifstream s(i->c_str());
			po::store(po::parse_config_file(s, fop), vm);
		}
	}

	// Parse stdin
	if (!isatty(STDIN_FILENO)) {
		po::store(po::parse_config_file(std::cin, fop), vm);
	}

	if (vm.count("version")) {
		std::cout << "fishFlow v0.2 - 2014/07/15" << std::endl;
		std::exit(0);
	}
	if (vm.count("help")) {
		std::cout << "Usage: " << argv[0] << " [options] [config_file] ...";
		std::cout << std::endl << std::endl;
		std::cout << "Synopsis: Compute velocity from videos of fish schools using optical flow";
		std::cout << std::endl << std::endl;
		std::cout << op << std::endl;
		std::exit(0);
	}
	po::notify(vm);
	if (!vm.count("data") && !vm.count("movie") && !vm["live"].as<bool>() && !vm.count("info")) {
		std::cout << "Fatal: no output was specified" << std::endl;
		std::exit(-1);
	}
	return vm;
}

void frameLogic(const po::variables_map& config, size_t& start, size_t& stop, size_t& step, size_t& count, size_t max_count) {
	start = config["frame.start"].as<int>();
	step = config["frame.step"].as<int>();
	stop = max_count;
	count = max_count;

	if (config.count("frame.stop")) {
		stop = config["frame.stop"].as<int>();
	}
	if (config.count("frame.count")) {
		count = config["frame.count"].as<int>();
	}
	if (start == 0) {
		std::cerr << "Fatal: frame start cannot be zero (frames start at one)" << std::endl;
		std::exit(-1);
	}
	if (step < 1) {
		std::cerr << "Fatal: frame step cannot be less than one" << std::endl;
		std::exit(-1);
	}
	if (start > stop) {
		std::cerr << "Fatal: frame start cannot be after frame stop" << std::endl;
		std::exit(-1);
	}
	if (stop > max_count) {
		std::cerr << "Fatal: frame stop is past the last frame" << std::endl;
		std::exit(-1);
	}
	if ((stop - start + 1) / step != count) {
		if (config.count("frame.stop") && config.count("frame.count")) {
			std::cout << "Fatal: both frame stop and count specified" << std::endl;
			std::exit(-1);
		} else if (config.count("frame.count")) {
			stop = start + (count - 1) * step;
		} else {
			count = (stop - start + 1) / step;
		}
	}
	if (start == stop) {
		std::cerr << "Fatal: not enough frames to process" << std::endl;
		std::exit(-1);
	}
	if (stop > max_count) {
		stop = max_count;
	}
}

void siginthandler(int param)
{
	std::cerr << std::endl;
	h5flow.flush();
	h5contour.flush();
	std::exit(-1);
}


const char* dots[] = {
	"⠀", "⠁", "⠂", "⠃", "⠄", "⠅", "⠆", "⠇", "⠈", "⠉", "⠊", "⠋", "⠌", "⠍", "⠎", "⠏",
	"⠐", "⠑", "⠒", "⠓", "⠔", "⠕", "⠖", "⠗", "⠘", "⠙", "⠚", "⠛", "⠜", "⠝", "⠞", "⠟",
	"⠠", "⠡", "⠢", "⠣", "⠤", "⠥", "⠦", "⠧", "⠨", "⠩", "⠪", "⠫", "⠬", "⠭", "⠮", "⠯",
	"⠰", "⠱", "⠲", "⠳", "⠴", "⠵", "⠶", "⠷", "⠸", "⠹", "⠺", "⠻", "⠼", "⠽", "⠾", "⠿",
	"⡀", "⡁", "⡂", "⡃", "⡄", "⡅", "⡆", "⡇", "⡈", "⡉", "⡊", "⡋", "⡌", "⡍", "⡎", "⡏",
	"⡐", "⡑", "⡒", "⡓", "⡔", "⡕", "⡖", "⡗", "⡘", "⡙", "⡚", "⡛", "⡜", "⡝", "⡞", "⡟",
	"⡠", "⡡", "⡢", "⡣", "⡤", "⡥", "⡦", "⡧", "⡨", "⡩", "⡪", "⡫", "⡬", "⡭", "⡮", "⡯",
	"⡰", "⡱", "⡲", "⡳", "⡴", "⡵", "⡶", "⡷", "⡸", "⡹", "⡺", "⡻", "⡼", "⡽", "⡾", "⡿",
	"⢀", "⢁", "⢂", "⢃", "⢄", "⢅", "⢆", "⢇", "⢈", "⢉", "⢊", "⢋", "⢌", "⢍", "⢎", "⢏",
	"⢐", "⢑", "⢒", "⢓", "⢔", "⢕", "⢖", "⢗", "⢘", "⢙", "⢚", "⢛", "⢜", "⢝", "⢞", "⢟",
	"⢠", "⢡", "⢢", "⢣", "⢤", "⢥", "⢦", "⢧", "⢨", "⢩", "⢪", "⢫", "⢬", "⢭", "⢮", "⢯",
	"⢰", "⢱", "⢲", "⢳", "⢴", "⢵", "⢶", "⢷", "⢸", "⢹", "⢺", "⢻", "⢼", "⢽", "⢾", "⢿",
	"⣀", "⣁", "⣂", "⣃", "⣄", "⣅", "⣆", "⣇", "⣈", "⣉", "⣊", "⣋", "⣌", "⣍", "⣎", "⣏",
	"⣐", "⣑", "⣒", "⣓", "⣔", "⣕", "⣖", "⣗", "⣘", "⣙", "⣚", "⣛", "⣜", "⣝", "⣞", "⣟",
	"⣠", "⣡", "⣢", "⣣", "⣤", "⣥", "⣦", "⣧", "⣨", "⣩", "⣪", "⣫", "⣬", "⣭", "⣮", "⣯",
	"⣰", "⣱", "⣲", "⣳", "⣴", "⣵", "⣶", "⣷", "⣸", "⣹", "⣺", "⣻", "⣼", "⣽", "⣾", "⣿"
};


int main(int argc, char* argv[]) {
	signal(SIGINT, siginthandler);

	H5File h5flow, h5contour;
	size_t total_points_contour = 0;
	H5::CompType xy_dtype;

	// Read config
	po::variables_map config;
	try {
		config = parse(argc, argv);
	} catch (const std::exception& e) {
		std::cerr << "Fatal: " << e.what() << std::endl;
		return -1;
	}

	// Load video
	cv::VideoCapture cap(config["input"].as<std::string>());
	if (!cap.isOpened()) {
		std::cerr << "Fatal: the input file cannot be open" << std::endl;
		return -1;
	}
	const int w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	const int h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	const int c = cap.get(CV_CAP_PROP_FRAME_COUNT);
	const double fps = cap.get(CV_CAP_PROP_FPS);

	if (config.count("info")) {
		std::cout << config["input"].as<std::string>() << ": ";
		std::cout << w << "x" << h << " @ " << fps << "fps - duration: ";
		std::cout << trunc(c / fps / 3600) << ":" << fmod(trunc(c / fps / 60), 60) << ":" << fmod(c / fps, 60);
		std::cout << " (" << c << " frames)" << std::endl;
		std::exit(0);
	}

	size_t start, stop, step, count;
	frameLogic(config, start, stop, step, count, c);

	// Compute background image
	std::string bgpath;
	cv::Mat im;
	MMat gm, bg;
	if (config.count("background")) {
		bgpath = config["background"].as<std::string>();
		cv::Mat tmp = cv::imread(bgpath, CV_LOAD_IMAGE_GRAYSCALE);
#ifdef USE_CL
		bg = tmp.getUMat(cv::ACCESS_READ);
#else
		bg = tmp;
#endif
	} else {
		bgpath = config["input"].as<std::string>() + ".background.jpg";
	}
	if (bg.empty()) {
		bg = MMat(h, w, CV_32FC1, cv::Scalar::all(0));
		std::cout << "Computing background image:" << std::endl;
		std::cout << "    0% (1/" << c << ")" << std::flush;
		for (int i = 0; cap.read(im); ++i) {
			cv::cvtColor(im, gm, CV_RGB2GRAY); // convert to grayscale
			cv::accumulate(gm, bg);
			std::cout << '\r' << dots[i%256] << ' ' << std::setw(3) << (i + 1) * 100 / c << "% (" << i+1 << "/" << c << ")" << std::flush;
		}
		cv::divide(bg, c, bg);
		bg.convertTo(bg, CV_8U);
		cv::imwrite(bgpath, bg);
		std::cout << std::endl;
	}

	const bool invert = config["invert"].as<bool>();
	if (invert) {
	    cv::subtract(255, bg, bg);
	}

	// Rewind
	cap.set(CV_CAP_PROP_POS_FRAMES, 0);

	// Skip frames instead of setting CV_CAP_PROP_POS_FRAMES to avoid issue with keyframes
	for (int k = 1; k < start; ++k) cap.grab();

	// Prepare output files
	const bool live = config["live"].as<bool>();
	const bool vid = config.count("movie");
	const bool data = config.count("data");
	if (live) cv::namedWindow("fishFlow live");
	cv::VideoWriter vw;
	if (vid) {
		std::string path = config["movie"].as<std::string>();
		if (path.empty()) {
			path = config["input"].as<std::string>() + ".flow.avi";
		}
		vw.open(path, CV_FOURCC('M','J','P','G'), fps, cv::Size(w, h));
		if (!vw.isOpened()) {
			std::cerr << "Fatal: the output video file cannot be open" << std::endl;
			return -1;
		}
	}
	const size_t gw = config["grid.width"].as<size_t>();
	const size_t gh = config["grid.height"].as<size_t>();
	Plot plot(gw, gh);

	if (data) {
		std::string path = config["data"].as<std::string>();
		if (path.empty()) {
			path = config["input"].as<std::string>() + ".flow.h5";
		}

		xy_dtype = H5::CompType(2 * sizeof(float));
		xy_dtype.insertMember("x", 0 * sizeof(float), H5::PredType::NATIVE_FLOAT);
		xy_dtype.insertMember("y", 1 * sizeof(float), H5::PredType::NATIVE_FLOAT);

		const hsize_t dims_flow[3] = {count-1, gw, gh};

		h5flow.init(path, true);
		h5flow.add_data_set<3>("velocity", xy_dtype, dims_flow);
		h5flow.add_data_set<3>("density", H5::PredType::NATIVE_UCHAR, dims_flow);

		const hsize_t dims_bp[2] = {3, 10};
		const hsize_t max_dims[2] = {3, H5S_UNLIMITED};
		const hsize_t chunk_dims[2] ={3, 10};

		h5flow.add_extendable_data_set<2>("boundaryPaths", H5::PredType::NATIVE_FLOAT, dims_bp, max_dims, chunk_dims);
	}

	// Compute density and optical flow
	std::cerr << "Computing density and optical flow:" << std::endl;
	const cv::Size size(gw, gh);
	MMat prev, next, mask, uv;
	MMat sd(gh, gw, CV_8UC1), suv(gh, gw, CV_32FC2);
	// Gunnar Farneback’s Optical Flow options
	const double pyr_scale = 0.5;
	const int levels = 2;
	const int winSize = 45;
	const int iterations = 4;
	const int poly_n = 7;
	const double poly_sigma = 1.5;
	int flags = cv::OPTFLOW_FARNEBACK_GAUSSIAN;
	std::cout << "    0% (1/" << count << ")" << std::flush;
	std::chrono::time_point<std::chrono::system_clock> time_start, time_end;
	for (size_t i = start, j = 0; i <= stop && cap.read(im); i += step, ++j) {
		time_start = std::chrono::system_clock::now();
		cv::cvtColor(im, gm, CV_RGB2GRAY); // convert to grayscale
		if(invert) {
		    cv::subtract(255, gm, gm);
		}

		// Compute density
		cv::subtract(bg, gm, gm);
		cv::subtract(255, gm, gm);
		prev = next;
		next = gm.clone();
		cv::threshold(gm, gm, 200, 255, cv::THRESH_BINARY);
		cv::GaussianBlur(gm, gm, cv::Size(95, 95), 0, 0);
		cv::addWeighted(gm, -4, gm, 0, 1024, gm);

		// Compute density mask
		cv::threshold(gm, mask, 40, 255, cv::THRESH_BINARY);

		// Compute optical flow
		if (i == start) continue; // requires two frames
		cv::calcOpticalFlowFarneback(prev, next, uv, pyr_scale, levels, winSize, iterations, poly_n, poly_sigma, flags);
		flags |= cv::OPTFLOW_USE_INITIAL_FLOW;

		if (live || vid) {
			cv::Mat gm_tmp, uv_tmp, mask_tmp;
			gm.copyTo(gm_tmp);
			uv.copyTo(uv_tmp);
			mask.copyTo(mask_tmp);
			cv::addWeighted(im, 0.5, color(gm_tmp), 0.5, 0, im);
			plot.plotVelocity(im, uv_tmp, mask_tmp);
		}

		if (live) {
			cv::imshow("fishFlow live", im);
			cv::waitKey(10);
		}

		if (vid) {
			vw.write(im);
		}

		if (data) {
			cv::resize(uv, suv, size);
			cv::flip(suv, suv, 0);
			cv::transpose(suv, suv);

			cv::resize(gm, sd, size);
			cv::flip(sd, sd, 0);
			cv::transpose(sd, sd);

			const hsize_t start_flow[3] = { j-1, 0, 0 };
			const hsize_t count_flow[3] = { 1, gw, gh };

#ifdef USE_CL
			h5flow.write<3>("velocity", suv.getMat(cv::ACCESS_READ).ptr(), start_flow, count_flow);
			h5flow.write<3>("density", sd.getMat(cv::ACCESS_READ).ptr(), start_flow, count_flow);
#else
			h5flow.write<3>("velocity", suv.ptr(), start, count);
			h5flow.write<3>("density", sd.ptr(), start, count);
#endif

			std::vector<std::vector<cv::Point> > contours;
			cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
			std::vector<cv::Point> largest_contour;
			for(size_t i = 0; i < contours.size(); i++) {
				std::vector<cv::Point> c = contours.at(i);
				if(c.size() > largest_contour.size())
					largest_contour = c;
			}

			size_t dataChunkStart = total_points_contour;
			const size_t n = largest_contour.size();
			total_points_contour += n;

			const hsize_t start_bp[2] = {0, dataChunkStart};
			const hsize_t count_bp[2] = {3, n};
			float data[3][n];
			for(int i = 0; i < n; i++) {
				data[0][i] = largest_contour.at(i).x;
				data[1][i] = largest_contour.at(i).y;
				data[2][i] = j;
			}
			h5contour.write<2>("boundaryPaths", &data, start_bp, count_bp);
		}

		time_end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = time_end-time_start;
		std::cout << '\r' << dots[j%256] << ' ' << std::setw(3) << (j + 1) * 100 / count << "% (" << j+1 << "/" << count << ")"
			  << " elapsed time: " << elapsed_seconds.count() << " fps: " << (1.f / elapsed_seconds.count()) << std::flush;

		// Skip frames instead of setting CV_CAP_PROP_POS_FRAMES to avoid issue with keyframes
		for (int k = 1; k < step; ++k) cap.grab();
	}
	std::cout << std::endl;
	if (live) cv::destroyAllWindows();

	return 0;
}
