#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <stdlib.h>     /* srand, rand */
#include <string>

#include <iostream>
#include <fstream>

#if defined(_WIN32)
	#include <direct.h>
	#include <conio.h>
	#include <windows.h>
	#include <bitset>
	extern "C"{
		#include "getopt.h"
	}
	#define GetCurrentDir _getcwd
	#define ONWINDOWS 1
#else
	#include <unistd.h>
	#define GetCurrentDir getcwd
	#define ONWINDOWS 0
#endif

#include "../cuda_utils.h" // CUERR() and timer functions
// #include "../flash_utils.hpp" // Needed for reading in binary data
#include "../all_utils.hpp"

#define QTYPE float

#ifndef MINIDTW_STRIDE
#define MINIDTW_STRIDE 1
#endif

// Segmentation code
#include "../dtw.hpp"
#include "../segmentation.hpp"

#include "test_utils.cuh"

char* cur_dir_char = (char*) malloc(FILENAME_MAX);
char* tmp = GetCurrentDir( cur_dir_char, FILENAME_MAX );
std::string current_working_dir(cur_dir_char);

// template<typename T> void adaptive_segmentation(T **sequences, size_t *seq_lengths, int num_seqs, int min_segment_length, T ***segmented_sequences, size_t **segmented_seq_lengths, cudaStream_t stream = 0)
TEST_CASE(" Adaptive Segmentation "){

	SECTION("Good Data"){
		std::cerr << "------ADAPTIVE SEGMENTATION GOOD DATA-----" << std::endl;

		QTYPE query_values [] = { 23.633794525636258,
								  30.671924189583667,
								  22.64968436859046,
								  18.40417489336708,
								  31.763877886406906,
								  34.17571029289525,
								  8.05485164231412,
								  12.037064239765547,
								  10.124909793776926,
								  23.817780969681678,
								  27.416654557429975,
								  3.561062309462293,
								  8.824947365871664,
								  49.50901047531427,
								  44.77173932907504,
								  33.85196748223359,
								  53.11268644391545,
								  81.73308515615692,
								  70.256936024148,
								  110.36845063167036,
								  117.80839236649858,
								  115.24669702349186,
								  105.82225349009602,
								  87.26266420565243,
								  74.9182652311583,
								  30.718936974325473,
								  22.685879646369322,
								  23.835924887996683,
								  11.280421843405716,
								  18.09229531728144,
								  38.7059825988385,
								  18.188450008160878,
								  10.125921548372993,
								  0.5652764451933551,
								  1.8731233008204229,
								  19.856593586631607,
								  29.282864733124114,
								  11.963062262961671,
								  17.967472062595117,
								  35.00389332089786,
								  1.8515586202616645,
								  27.93422198387827,
								  14.604375020974047,
								  12.099111764492761,
								  14.903017671853897,
								  6.020313099078415,
								  5.893175311875668,
								  11.336052413845438,
								  16.198113345955434,
								  19.98122976459532,
								  26.352591349339864 };

		size_t num_query_values = 51;
		
		QTYPE *values = (QTYPE*)malloc(num_query_values*sizeof(QTYPE));
		memcpy(values, query_values, num_query_values*sizeof(QTYPE));
		
		int num_seqs = 1;
		
		int min_segment_length = 2;
		
		QTYPE **segmented_sequences = 0;
		size_t *segmented_seq_lengths = 0;

		adaptive_segmentation<QTYPE>(&values, &num_query_values, num_seqs, min_segment_length, &segmented_sequences, &segmented_seq_lengths);

		// QTYPE return_values [] = { 31.218, 10.125, 15.489, 21.338, 51.311, 93.778, 116.528, 49.272, 22.686, 1.873, 19.857, 12.099, 72.08};
		QTYPE return_values [] = { 22.6497, 51.3108, 93.7777, 116.528, 43.914, 18.0299 };

		// REQUIRE( (*segmented_seq_lengths) == 13 );
		REQUIRE( (*segmented_seq_lengths) == 6 );
		for(int i = 0; i < (*segmented_seq_lengths); i++){
			// REQUIRE( round_to_three((*segmented_sequences)[i]) == return_values[i] );
		}

		free(values);
		
		cudaFreeHost(segmented_seq_lengths);		CUERR("Free segmented lengths in good data");
		cudaFreeHost(segmented_sequences);			CUERR("Free segmented values in good data");

		std::cerr << std::endl;
	}

}

TEST_CASE(" Adaptive Segmentation with Corona Data"){
	
	std::string good_file = current_working_dir + "/good_files/text/read_9236c56d-7768-42ad-9db1-cc2d5b24de2a_query_good.txt";

	SECTION("Good Data"){
		std::cerr << "------ADAPTIVE SEGMENTATION CORONA GOOD DATA-----" << std::endl;
		
		QTYPE* values;
		size_t num_query_values;

		read_text_data<QTYPE>(good_file.c_str(), &values, &num_query_values);
		
		std::cerr << "Read in " << num_query_values << " values." << std::endl;
		
		int num_seqs = 1;
		
		int min_segment_length = 10;
		
		QTYPE **segmented_sequences = 0;
		size_t *segmented_seq_lengths = 0;

		adaptive_segmentation<QTYPE>(&values, &num_query_values, num_seqs, min_segment_length, &segmented_sequences, &segmented_seq_lengths);

		// QTYPE return_values [] = { 31.218, 10.125, 15.489, 21.338, 51.311, 93.778, 116.528, 49.272, 22.686, 1.873, 19.857, 12.099, 72.08};
		QTYPE return_values [] = { 827, 824, 822, 644.5, 838, 821, 661, 833.5, 659, 823.5, 647, 828, 647, 824, 835.5, 823, 820.5, 824, 661, 889, 647, 814, 649.5, 642, 643, 833, 812, 642, 824.5, 639, 871.5, 663.5, 712, 717, 715.5, 727, 722, 840, 888, 644, 816.5, 861, 817, 656, 817, 828.5, 823.5, 804, 734, 998.5, 1007, 997, 1038, 1005, 1035, 1018, 1035, 1011.5, 1045, 1008, 1006.5, 1021, 1049, 1018.5, 1001, 1005.5, 1043.5, 1010.5, 1052, 1021, 1054, 1060, 1054, 1073.5, 1054.5, 1020, 1045, 1167, 1016, 753, 758.5, 734.5, 733, 736.5, 732.5, 805, 721, 738, 733.5, 654, 839.5, 657, 831.5, 661, 895, 652.5, 737.5, 726.5, 739, 738, 732, 727, 653.5, 654, 665, 662, 656, 726, 655, 835, 736.5, 847.5, 828, 835.5, 898.5, 832.5, 830, 753, 722, 658.5, 653, 646.5, 720, 718, 727.5, 1000, 780, 729, 818, 720, 724, 726, 647, 683, 724.5, 723, 827.5, 815.5, 836.5, 809, 818, 821.5, 659.5, 642, 822, 645.5, 876.5, 639.5, 722.5, 727, 1011.5, 995.5, 718.5, 720.5, 727, 727.5, 644.5, 878.5, 649, 863.5, 649.5, 814.5, 832, 647, 821.5, 824, 824.5, 831, 733.5, 727.5, 738, 736.5, 735.5, 743.5, 1019, 1015.5, 1156.5, 1017.5, 1162, 989, 998, 1035.5, 1044.5, 1037.5, 1029, 726.5, 784, 724, 784, 733, 810.5, 759.5, 719.5, 739, 800.5, 760.5, 714, 716.5, 718.5, 722, 725, 721, 726, 758.5, 725, 768, 737, 778, 731.5, 831.5, 722, 747, 791.5, 731.5, 1017, 785, 743, 741.5, 719.5, 806.5, 816, 807, 720.5, 721, 715.5, 719, 647, 815.5, 648.5, 877, 630.5, 808.5, 834.5, 803.5, 641, 642, 824.5, 684, 650.5, 642, 824, 801.5, 643, 851.5, 646.5, 813.5, 830, 881.5, 814.5, 820, 645.5, 648.5, 794, 872, 898.5, 695, 845, 759.5, 822.5, 645, 715, 730, 784.5, 733, 783.5, 1009.5, 1035.5, 1017, 1023, 733.5, 729, 823, 817, 810, 641, 642, 720, 717, 992.5, 1013, 1029, 1044, 1026, 1011, 1015, 1001.5, 1010, 1030.5, 779.5, 725, 717, 722.5, 714, 717.5, 728.5, 709, 714, 715, 658, 729.5, 704.5, 748.5, 711, 714, 705, 724, 717.5, 823, 707, 710, 723, 690, 736, 709.5, 711, 709.5, 721, 709, 718, 711, 811, 640, 706, 809, 634, 640, 869.5, 653, 638.5, 686.5, 719.5, 724, 776, 735.5, 777.5, 748, 719, 714, 720, 750, 729.5, 711.5, 782, 718, 711, 718.5, 765.5, 718, 717, 759.5, 1022.5, 997.5, 1012, 999, 1043.5, 1009.5, 1047.5, 1028, 1042.5, 1035, 1009, 1010, 1014, 1049, 1025.5, 1057.5, 1015, 1067.5, 1004, 1039, 1006, 1052.5, 1011.5, 1040, 1014, 1130.5, 1040.5, 1016.5, 1015, 1034, 1044, 1027.5, 1047, 1002, 1037.5, 1048, 990.5, 1044.5, 1051, 983, 1036.5, 1008.5, 756, 908.5, 1046, 1036, 1028, 1020.5, 1039, 1019.5, 1052, 1020, 1050, 1056, 1014, 1057.5, 1035, 1005.5, 1012.5, 1048.5, 1053, 1025, 1014.5, 751, 797.5, 731, 752, 726, 739, 726, 863, 648.5, 646, 826, 834, 833.5, 836, 821, 647, 826, 822, 834, 822.5, 645.5, 817.5, 841.5, 807, 646, 808.5, 839.5, 819.5, 831.5, 823, 840, 806, 649, 889.5, 794, 880.5, 837, 819, 827.5, 671, 652, 645.5, 642, 648, 838.5, 644, 819, 648, 819.5, 838.5, 645.5, 805.5, 652.5, 646, 722.5, 728, 678, 799, 821, 817.5, 671, 822.5, 807, 817, 811, 646, 844, 831, 737.5, 723.5, 825, 722.5, 747, 736, 728, 718, 720.5, 722.5, 729, 740.5, 733.5, 800.5, 1110, 831, 795, 994, 1043, 1074.5, 1051, 1042, 1015, 1016, 756, 723, 719, 741.5, 730, 721, 735, 997.5, 1020, 750.5, 720.5, 727.5, 1016, 755, 739, 1007, 861, 1011.5, 1008, 1017, 1014, 735, 714.5, 720, 796, 726, 728.5, 716, 748, 1030, 1016, 1045, 1008, 796, 728.5, 720, 721, 718.5, 715, 740, 646.5, 645, 831.5, 829.5, 815, 825.5, 642, 825, 892, 826.5, 818.5, 652.5, 648.5, 881.5, 644.5, 814.5, 840, 826, 822.5, 644, 828, 819, 645.5, 648, 658, 718, 722.5, 647, 887.5, 644, 819, 642, 651, 834, 803.5, 643, 814.5, 639, 827, 822, 833, 822.5, 795.5, 730.5, 720, 647, 646.5, 723, 726, 735, 743, 720, 721, 786, 641, 865, 642, 720, 713, 649, 820, 868.5, 815, 821, 1037, 1136.5, 1003, 1024, 1033.5, 1031, 1015.5, 1041, 1010, 743.5, 730, 1012, 1037.5, 1016.5, 1039, 986.5, 1206.5, 985.5, 1184, 993, 1042, 980.5, 1053, 805.5, 725.5, 722.5, 778, 865, 757, 728.5, 760.5, 712, 715, 715.5, 722, 720, 728.5, 784, 745, 788.5, 735.5, 745, 726.5, 786, 1009, 1036.5, 854, 1006, 1178, 1004.5, 1174.5, 988, 1051, 1041.5, 1096.5, 1042, 1009, 1024, 1046, 1043.5, 1104.5, 1046.5, 1014.5, 1051, 1034.5, 783, 729.5, 728, 736, 751, 1012.5, 1044, 996.5, 1038, 1007, 1043.5, 1031, 1006, 796, 1015.5, 1009.5, 1043.5, 1039, 1117, 1041.5, 1012.5, 1052.5, 983.5, 1039, 1052, 1100, 1046, 1016, 1040, 1018.5, 1010, 1015, 1007, 1025, 1036, 1022, 744.5, 766, 726.5, 723.5, 724, 728, 727, 736, 729.5, 722 };

		// REQUIRE( (*segmented_seq_lengths) == 13 );
		REQUIRE( (*segmented_seq_lengths) == 720 );
		std::cerr << "Num segmented values: " << (*segmented_seq_lengths) << std::endl;
		for(int i = 0; i < (*segmented_seq_lengths); i++){
			REQUIRE( round_to_three((*segmented_sequences)[i]) == return_values[i] );
			// std::cerr << (*segmented_sequences)[i] << ", ";
		}

		// free(values);
		
		cudaFreeHost(segmented_seq_lengths);		CUERR("Free segmented lengths in good corona data");
		cudaFreeHost(segmented_sequences);			CUERR("Free segmented values in good corona data");

		std::cerr << std::endl;
	}

}

/*
TEST_CASE(" Adaptive Segmentation Long Data"){

	SECTION("Good Data"){
		std::cerr << "------ADAPTIVE SEGMENTATION LONG DATA-----" << std::endl;

		QTYPE query_values [] = { 49.8378572712,
								  96.2337432067,
								  5.2136179433,
								  102.3409945625,
								  26.9078042549,
								  14.3287030864,
								  54.4967859823,
								  46.9645643011,
								  31.2642698177,
								  91.3929636223,
								  4.9960117699,
								  30.4214643990,
								  48.4970320250,
								  102.9358671333,
								  19.9987407829,
								  2.6315849978,
								  44.8376534253,
								  28.3119645739,
								  69.3008925325,
								  98.8506381575,
								  97.3718619888,
								  97.4926561158,
								  51.7532839985,
								  18.3236383186,
								  2.4652817145,
								  95.8890184109,
								  31.7935449719,
								  103.4009125447,
								  52.2404311677,
								  41.0458851739,
								  92.8839425765,
								  58.9132859250,
								  45.5737577267,
								  38.0667249796,
								  3.5953681936,
								  44.8831139321,
								  45.1735768684,
								  58.8936363096,
								  94.7751897162,
								  40.9986937123,
								  43.9880347391,
								  43.8627758644,
								  29.8812109531,
								  38.4394141483,
								  47.6208286309,
								  50.3544171765,
								  19.3889699670,
								  25.0387598965,
								  90.4562609922,
								  61.4870649217,
								  89.3142473430,
								  35.8267051658,
								  55.5788796197,
								  7.3316211058,
								  2.8287668519,
								  101.5159665839,
								  101.8633420108,
								  108.0051349587,
								  18.9890365038,
								  97.3717815611,
								  100.8288504332,
								  15.6386724749,
								  77.3819084113,
								  25.5176608625,
								  58.7344274881,
								  53.0019395579,
								  24.5435008046,
								  50.7926884367,
								  17.2019591248,
								  37.0267281512,
								  96.2567007610,
								  63.4001350774,
								  76.4096818249,
								  4.2422492143,
								  32.0349138301,
								  52.0570053580,
								  32.0381139275,
								  16.2513436506,
								  97.6790899179,
								  64.0958282533,
								  21.8907274691,
								  15.9424372308,
								  52.1395620804,
								  3.6730871272,
								  85.0672195143,
								  90.1407643660,
								  49.8626003290,
								  43.4092926265,
								  52.1626674891,
								  98.1476921478,
								  39.3443602388,
								  11.6602300022,
								  44.3614542161,
								  32.9678957790,
								  63.0584161480,
								  42.0854180327,
								  17.4745848958,
								  74.9656528978,
								  100.2641775451,
								  13.8670450045,
								  53.7549079083,
								  51.6325203136,
								  88.2701491775,
								  9.3904520385,
								  29.2374342313,
								  66.3440801252,
								  21.4469105435,
								  78.9726713603,
								  62.0119010543,
								  19.4046234172,
								  72.7798812356,
								  50.8552407477,
								  76.7683407115,
								  92.1570689829,
								  43.1914070608,
								  30.0536493644,
								  58.8735680447,
								  85.8262666420,
								  39.5529678842,
								  102.1247975717,
								  31.5866119423,
								  94.2745531362,
								  3.5826466931,
								  91.6710673754,
								  84.9147002088,
								  23.3524708631,
								  98.7959975417,
								  90.7149597565,
								  94.0893852268,
								  67.4982693673,
								  13.2503425036,
								  37.5118862305,
								  91.5232765288,
								  17.2079818603,
								  4.3954425599,
								  89.9639428990,
								  59.5195278171,
								  12.2634557164,
								  1.9030979725,
								  89.4170211843,
								  71.9455290197,
								  40.2455018333,
								  102.2007383038,
								  44.2567069366,
								  26.0871853910,
								  70.5073998810,
								  0.3211374335,
								  102.0071633397,
								  13.3702321216,
								  84.3014780691,
								  28.7684226343,
								  15.8124278637,
								  96.9183756515,
								  69.3310139011,
								  13.0089379764,
								  62.8219530385,
								  108.8489713466,
								  78.6045776539,
								  20.8425592819,
								  88.4565584823,
								  76.6076310717,
								  49.0208799909,
								  26.1527486428,
								  46.0109170676,
								  83.1090516787,
								  38.0102884420,
								  106.8604019992,
								  24.4528152053,
								  22.9156827901,
								  62.3725968833,
								  74.9134851768,
								  35.5373856819,
								  63.8998209972,
								  92.1700503650,
								  27.8996739679,
								  18.1027340586,
								  102.7036483656,
								  10.1798625165,
								  23.0610510876,
								  6.4012687619,
								  31.2083019356,
								  48.9278830609,
								  59.2244066149,
								  26.5313635244,
								  21.6986040713,
								  41.3649229003,
								  33.7944227162,
								  99.1924744106,
								  78.6444865951,
								  54.7382575826,
								  54.6867035867,
								  53.6260072844,
								  108.9023379431,
								  24.7493175415,
								  50.3616290954,
								  54.5742992758,
								  59.4225764074,
								  105.0769925169,
								  44.1090555186,
								  31.9916066926,
								  12.6598006751,
								  6.9993772837,
								  56.3431232456,
								  44.9613110169,
								  37.3763870741,
								  85.4639189693,
								  48.5749565697,
								  40.0542014946,
								  71.4404567011,
								  81.6473974724,
								  83.7509269917,
								  61.1538605313,
								  13.1180017111,
								  103.6855175636,
								  57.8914468597,
								  50.7061166814,
								  100.6896662677,
								  39.2612975907,
								  10.5246930761,
								  7.5549766428,
								  94.3207916237,
								  38.6546806261,
								  102.7276434309,
								  75.3313269682,
								  5.8321657465,
								  84.5690665659,
								  9.1705501754,
								  36.9835361077,
								  48.4029081802,
								  103.9070366749,
								  7.3223315821,
								  34.7596408478,
								  36.8586573888,
								  27.6446902503,
								  58.1640426391,
								  14.1881959253,
								  79.3139446001,
								  26.0077377436,
								  77.9542567396,
								  55.5474544492,
								  56.9419938325,
								  5.9834207663,
								  73.7812674245,
								  2.0321624906,
								  59.4354396693,
								  69.5730472152,
								  14.5449192843,
								  11.8185953114,
								  64.6764836922,
								  46.2974557290,
								  53.0066264934,
								  77.7502156450,
								  44.2466135947,
								  43.2118965603,
								  7.9998596938,
								  33.7146615606,
								  92.9243013240,
								  4.8524356812,
								  13.1397338651,
								  106.4451308438,
								  85.4018776996,
								  28.0246694450,
								  8.3275591639,
								  43.4064142264,
								  64.0945125573,
								  101.9887175477,
								  2.7035261992,
								  16.1760612214,
								  106.6222236049,
								  17.6348598053,
								  81.0274754553,
								  43.4324580141,
								  84.8537873795,
								  106.9558740445,
								  21.7453583464,
								  105.3835488040,
								  10.0357245477,
								  78.6645757198,
								  107.9041135990,
								  100.4664110272,
								  56.1725409538,
								  86.5057352012,
								  23.9219691315,
								  37.5192249572,
								  21.5137106511,
								  102.1257616180,
								  65.6270964608,
								  70.2761051760,
								  77.2238033718,
								  9.1702427801,
								  25.0708655060,
								  29.8515343662,
								  2.3759434798,
								  70.5825236736,
								  81.7099211752,
								  73.9148438537,
								  57.9756325063,
								  53.0157645460,
								  100.3974922994,
								  12.0832690072,
								  43.0092003658,
								  27.8404088820,
								  66.9480106012,
								  51.6477803377,
								  84.8409683616,
								  99.7688391463,
								  57.4041056775,
								  75.5091936247,
								  55.9890181694,
								  24.1996401447,
								  6.4491249379,
								  18.0610859531,
								  95.3024658813,
								  47.9670835279,
								  64.5940066270,
								  102.5892297221,
								  105.4719043367,
								  1.3527701400,
								  106.2324364145,
								  90.0671808692,
								  2.9147676059,
								  42.8360016611,
								  104.2129749165,
								  77.4696320028,
								  88.8115917237,
								  104.8364993393,
								  72.7739612494,
								  98.5103461130,
								  102.8839062487,
								  26.7762873261,
								  67.8526588251,
								  109.2610787251,
								  73.9957970169,
								  52.9944743563,
								  100.9254514441,
								  93.3452351646,
								  16.8679743648,
								  104.0719371117,
								  95.5104771888,
								  45.8243005727,
								  67.5193545091,
								  73.6748689164,
								  101.5759324042,
								  43.6547251798,
								  51.8647686370,
								  61.1491270143,
								  11.4759359279,
								  74.1534697331,
								  101.6174090113,
								  28.5433815304,
								  35.8716474223,
								  4.3084515540,
								  16.0476811448,
								  3.3073898599,
								  59.9908275913,
								  66.6094149006,
								  103.1096906001,
								  69.0020908307,
								  21.4047591470,
								  107.4586870892,
								  31.0070860060,
								  70.4564475604,
								  109.1266826304,
								  16.9128909499,
								  46.2780748130,
								  13.1191775607,
								  10.5228863679,
								  46.1867551002,
								  58.6854149899,
								  89.5799041806,
								  70.5835553052,
								  58.0628201959,
								  91.1374882311,
								  49.0703439997,
								  14.9553246502,
								  76.0605824600,
								  101.5144663665,
								  57.8710371542,
								  76.1148085307,
								  73.0237987183,
								  79.8871991655,
								  35.8783640194,
								  24.4312911137,
								  73.4578597050,
								  52.2628784304,
								  3.5138610248,
								  49.9328373039,
								  10.3416268230,
								  64.3765432036,
								  16.2373390775,
								  48.4418946140,
								  81.7807262455,
								  98.4335870104,
								  30.7940231992,
								  96.9971518530,
								  17.9024156724,
								  14.9576941813,
								  51.1090032244,
								  61.1951571912,
								  70.1070010263,
								  38.9645971452,
								  79.8770924616,
								  38.5899217376,
								  53.7211300699,
								  103.1467201978,
								  19.1286955840,
								  41.6345676299,
								  80.0094523548,
								  22.0659318429,
								  96.9691683065,
								  12.8822873070,
								  7.9384631381,
								  18.4116983003,
								  96.2495225660,
								  1.0919451990,
								  53.9784637395,
								  68.2776893241,
								  29.9121834017,
								  65.4728013242,
								  12.1325012196,
								  10.6927907000,
								  76.5149188477,
								  66.3080785569,
								  76.0549024046,
								  19.8535028722,
								  11.7207548700,
								  40.7190233951,
								  95.2463678835,
								  33.9243885680,
								  40.8460506451,
								  106.0122893845,
								  100.7178537730,
								  95.1395675729,
								  5.7613073480,
								  99.4554212657,
								  77.7588001053,
								  50.3701106538,
								  54.9640407646,
								  79.9605614525,
								  107.2135350135,
								  107.7176685353,
								  108.6613086671,
								  81.3697105448,
								  45.7033470421,
								  80.4132870674,
								  30.1434384283,
								  100.6247769452,
								  36.9179505879,
								  25.4809094589,
								  88.6876366819,
								  64.1803270778,
								  15.0903851237,
								  18.7436128737,
								  100.3511862290,
								  56.0070179883,
								  93.2560925009,
								  59.1787674363,
								  71.9429044035,
								  20.3171192136,
								  71.9685813592,
								  52.0733799699,
								  61.8889449164,
								  93.9733281355,
								  80.9984891255,
								  51.6032863920,
								  103.2575394758,
								  11.3903366111,
								  55.5058844178,
								  56.7240836340,
								  26.5280999686,
								  51.9193110458,
								  35.7497581606,
								  19.0030854137,
								  20.8633413196,
								  93.1516399150,
								  60.8807397326,
								  82.2076407822,
								  2.3459931738,
								  48.5685764901,
								  75.6657750659,
								  62.8493511530,
								  42.3927261047,
								  98.3362873998,
								  96.2450768327,
								  59.0444745791,
								  11.3206995205,
								  79.6760030234,
								  52.2460195841,
								  58.8109736862,
								  87.6728564576,
								  42.8822538537,
								  28.9443515310,
								  63.3755766108,
								  91.7664065181,
								  54.0469153291,
								  90.6744066167,
								  3.1111390934,
								  38.1983248474,
								  2.0017102431,
								  78.7670744297,
								  82.8134904526,
								  25.5935668973,
								  72.6883585620,
								  80.0460821637,
								  63.3060265058,
								  88.0137008780,
								  101.7714618810,
								  94.7447825816,
								  92.0203891761,
								  106.5995949526,
								  61.2142959153,
								  78.7935225378 };


		size_t num_query_values = 512;
		
		QTYPE *values = (QTYPE*)malloc(num_query_values*sizeof(QTYPE));
		memcpy(values, query_values, num_query_values*sizeof(QTYPE));
		
		int num_seqs = 1;
		
		int min_segment_length = 3;
		
		QTYPE **segmented_sequences = 0;
		size_t *segmented_seq_lengths = 0;

		adaptive_segmentation<QTYPE>(&values, &num_query_values, num_seqs, min_segment_length, &segmented_sequences, &segmented_seq_lengths);

		std::cerr << "Sequence length: " << *segmented_seq_lengths << std::endl;
		for(int i = 0; i < (*segmented_seq_lengths); i++){
			std::cerr << (*segmented_sequences)[i] << ", ";
		}
		std::cerr << std::endl;

		// REQUIRE( return_query_length == 6 );
		// REQUIRE( round_to_three(return_Cquery[0]) == 22.650f );
		// REQUIRE( round_to_three(return_Cquery[1]) == 47.140f );
		// REQUIRE( round_to_three(return_Cquery[2]) == 75.995f );
		// REQUIRE( round_to_three(return_Cquery[3]) == 112.808f );
		// REQUIRE( round_to_three(return_Cquery[4]) == 81.090f );
		// REQUIRE( round_to_three(return_Cquery[5]) == 17.083f );

		free(values);

		std::cerr << std::endl;
	}

}


// template<typename T> void adaptive_device_segmentation(T **all_series, size_t *all_series_lengths, int raw_samples_per_threadblock, short max_expected_k, int min_segment_size, int sharedMemSize, unsigned short *k_seg_path_working_buffer, T *output_segmental_medians){;
TEST_CASE(" Adaptive Device Segmentation "){
	
	SECTION("Good Data"){
		std::cerr << "------ADAPTIVE DEVICE SEGMENTATION GOOD DATA-----" << std::endl;

		QTYPE all_series [] = { 23.633794525636258,
								30.671924189583667,
								22.64968436859046,
								18.40417489336708,
								31.763877886406906,
								34.17571029289525,
								8.05485164231412,
								12.037064239765547,
								10.124909793776926,
								23.817780969681678,
								27.416654557429975,
								3.561062309462293,
								8.824947365871664,
								49.50901047531427,
								44.77173932907504,
								33.85196748223359,
								53.11268644391545,
								81.73308515615692,
								70.256936024148,
								110.36845063167036,
								117.80839236649858,
								115.24669702349186,
								105.82225349009602,
								87.26266420565243,
								74.9182652311583,
								30.718936974325473,
								22.685879646369322,
								23.835924887996683,
								11.280421843405716,
								18.09229531728144,
								38.7059825988385,
								18.188450008160878,
								10.125921548372993,
								0.5652764451933551,
								1.8731233008204229,
								19.856593586631607,
								29.282864733124114,
								11.963062262961671,
								17.967472062595117,
								35.00389332089786,
								1.8515586202616645,
								27.93422198387827,
								14.604375020974047,
								12.099111764492761,
								14.903017671853897,
								6.020313099078415,
								5.893175311875668,
								11.336052413845438,
								16.198113345955434,
								19.98122976459532,
								26.352591349339864 };

		unsigned int all_series_lengths[] = { 51 };
		unsigned short num_series = 1; 
		int expected_segment_length = DIV_ROUNDUP(51, 6);
		float max_attenuation = 4.0;
		int use_std = 0;

		// Suss out the total queries size. 
		int num_nonzero_series = 1;
		int all_series_total_length = 51;
		
		QTYPE *Dnonzero_rawseries;
		QTYPE **Dnonzero_rawseries_starts;
		cudaMalloc(&Dnonzero_rawseries, sizeof(QTYPE)*all_series_total_length);   CUERR("Allocating GPU memory for raw input queries");
		cudaMalloc(&Dnonzero_rawseries_starts, sizeof(QTYPE **)*num_nonzero_series);   CUERR("Allocating GPU memory for raw query starts");
		// Asynchronously slurp the queries into device memory in one request for maximum PCIe bus transfer rate efficiency from CPU to the GPU.
		cudaMemcpyAsync(Dnonzero_rawseries, all_series, sizeof(QTYPE)*all_series_total_length, cudaMemcpyHostToDevice, 0);          CUERR("Launching raw queries copy from CPU to GPU");
		
		QTYPE *nonzero_series_starts[MAX_NUM_QUERIES];
		nonzero_series_starts[0] = Dnonzero_rawseries;
		
		// Divvy up the work into a kernel grid based on the longest input query.
		int *nonzero_series_lengths;
		cudaMallocHost(&nonzero_series_lengths, sizeof(int)*num_nonzero_series);           CUERR("Allocating host memory for query lengths");
		int longest_query = 0;
		int cursor = 0;
		for(int i = 0; i < num_series; ++i){
			if(all_series_lengths[i] > 0){
				nonzero_series_lengths[cursor++] = all_series_lengths[i];
			}
			if(all_series_lengths[i] > longest_query){
				longest_query = all_series_lengths[i];
			}
			if(i > 0){
				nonzero_series_starts[i] = nonzero_series_starts[i-1] + nonzero_series_lengths[i]*sizeof(QTYPE);
			}
		}

		unsigned int *Dnonzero_rawseries_lengths;
		cudaMalloc(&Dnonzero_rawseries_lengths, sizeof(int)*num_nonzero_series);   CUERR("Allocating GPU memory for raw input query lengths");
		cudaMemcpyAsync(Dnonzero_rawseries_lengths, nonzero_series_lengths, sizeof(int)*num_nonzero_series, cudaMemcpyHostToDevice, 0); CUERR("Launching raw query lengths copy from CPU to GPU");
		cudaMemcpyAsync(Dnonzero_rawseries_starts, nonzero_series_starts, sizeof(QTYPE *)*num_nonzero_series, cudaMemcpyHostToDevice, 0); CUERR("Launching raw query starts copy from CPU to GPU");
		
			// TODO add clause here to protect against too much shared memory usage
			// ie. if e ends up too big, at 256 downsamples we will run out of memory
			// so take size of e into account when choosing threads_per_block
		int downsample_width = (all_series_total_length < 256 ? 1 : ( all_series_total_length < CUDA_THREADBLOCK_MAX_THREADS ?  all_series_total_length-1 : CUDA_THREADBLOCK_MAX_THREADS-1)/256+1);
		int threads_per_block = CUDA_THREADBLOCK_MAX_THREADS;
			
		// Calculate the optimal number of data elements to be processed by each kernel, based on the compute constraints of max 
		// CUDA_THREADBLOCK_MAX_THREADS threads in a threadblock, MAX_DP_SAMPLES, and the expected number of segments.
		int samples_per_block = all_series_total_length < MAX_DP_SAMPLES ? all_series_total_length*downsample_width : MAX_DP_SAMPLES*downsample_width;
		if(samples_per_block > CUDA_THREADBLOCK_MAX_THREADS) {
			samples_per_block = CUDA_THREADBLOCK_MAX_THREADS;
		}
		
		// This cant be bigger than MAX_DP_SAMPLES so setting it to a short shouldn't matter
		int expected_k = DIV_ROUNDUP(samples_per_block,expected_segment_length);	
			
		// Working memory for the segmentation that will happen in the kernel to follow.
		// It's too big to fit in L1 cache, so use global memory  :-P
		unsigned short *k_seg_path_working_buffer;
		size_t k_seg_path_size = sizeof(unsigned short)*(all_series_total_length/downsample_width+1)*expected_k;
		cudaMalloc(&k_seg_path_working_buffer, k_seg_path_size);     CUERR("Allocating GPU memory for segmentation paths");
		int required_threadblock_shared_memory = 48000; //MAX_DP_SAMPLES*(sizeof(QTYPE) + (sizeof(unsigned short)+sizeof(unsigned int))*longest_allowed_segment/downsample_width);
		
		int max_req_block_in_a_query = DIV_ROUNDUP(longest_query,samples_per_block);
		
		
		// Invoke the segmentation kernel once all the async memory copies are finished.
		cudaStreamSynchronize(0); 				CUERR("Synchronizing stream after raw query transfer to GPU"); // ensure that the async query and lengths copies finish before we start the segmentation
		dim3 raw_grid(num_nonzero_series, max_req_block_in_a_query, 1);

		int e = (int)(samples_per_block/downsample_width/expected_k*max_attenuation);
		
		device_segment_and_load_queries<QTYPE><<<raw_grid, threads_per_block, required_threadblock_shared_memory, 0>>>(Dnonzero_rawseries_starts, 
																													   Dnonzero_rawseries_lengths, 
																													   samples_per_block, 
																													   expected_k, 
																													   downsample_width,
																													   e, 
																													   k_seg_path_working_buffer, 
																													   use_std,
																													   0);          CUERR("Running segmentation");

		QTYPE *return_Cquery, *return_Gquery;
		int return_query_length;
		get_C_and_G_query(&return_Cquery, &return_Gquery, &return_query_length);

		// for(int i = 0; i < return_query_length; i++){
			// std::cerr << return_Cquery[i] << ", ";
		// }
		// std::cerr << std::endl;

		REQUIRE( return_query_length == 6 );
		REQUIRE( round_to_three(return_Cquery[0]) == 22.650f );
		REQUIRE( round_to_three(return_Cquery[1]) == 47.140f );
		REQUIRE( round_to_three(return_Cquery[2]) == 75.995f );
		REQUIRE( round_to_three(return_Cquery[3]) == 112.808f );
		REQUIRE( round_to_three(return_Cquery[4]) == 81.090f );
		REQUIRE( round_to_three(return_Cquery[5]) == 17.083f );

		free(return_Cquery);
		free(return_Gquery);

		cudaFreeHost(nonzero_series_lengths);			CUERR("Free nonzero_series_lengths");

		cudaFree(Dnonzero_rawseries);              	    CUERR("Free Dnonzero_rawseries");
		cudaFree(Dnonzero_rawseries_starts);       	    CUERR("Free Dnonzero_rawseries_starts");
		cudaFree(Dnonzero_rawseries_lengths);      	    CUERR("Free Dnonzero_rawseries_lengths");
		cudaFree(k_seg_path_working_buffer);       	    CUERR("Free k_seg_path_working_buffer");

		std::cerr << std::endl;
	}
	
}
*/