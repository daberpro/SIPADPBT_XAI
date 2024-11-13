
int main(uint32_t argc, char* argv[]){

    if(argc <= 1){
        std::cout << "The program need a model to work!\n";
        std::cout << "using XAI -h or XAI -help, to run help command\n";
        return EXIT_SUCCESS;
    }

    if(argc > 1 && (strcmp(argv[1],"-h") == 0 || strcmp(argv[1],"-help") == 0 )){
        std::cout << "To using this program run \"XAI <Model File> <Symptoms Data Index>\"\n";
        std::cout << "example : \"XAI durian-model.bin durian\", and remember do not insert space after <Symptoms Data Index>\n";
        std::cout << "but enter directly\n";
        return EXIT_SUCCESS;
    }

    std::string model_path = argv[1];
    std::string currrent_path = std::filesystem::current_path().string();
    std::ifstream index_data("index data.json");
    nlohmann::json json_data = nlohmann::json::parse(index_data);

    mlpack::FFN<mlpack::CrossEntropyError> model;
    mlpack::data::Load(model_path,"model",model);

    arma::mat input_data;
    uint32_t input_col_size = 0, output_row_size = 0;
    for(auto& element: json_data[argv[2]]["gejala"]){
        input_col_size++;
    }
    for(auto& element: json_data[argv[2]]["penyakit"]){
        output_row_size++;
    }
    input_data.reshape(1,input_col_size);

    uint32_t colWidth = 50;
    uint32_t cli_input, index = 0;
    for(auto& element: json_data[argv[2]]["gejala"]){
        std::cout << std::left 
        << std::setw(colWidth) << element.template get<std::string>()
        << " : ";
        std::cin >> cli_input;
        std::cin.ignore();
        input_data(0,index) = cli_input;
        index++;
    }

    input_data = input_data.t();
    arma::mat output_data;
    model.Predict(input_data,output_data);

    std::cout << "\nResult Prediction \n";
    std::cout << "==========================\n";
    for(uint32_t i = 0; i < output_row_size; i++){
        std::cout << std::left << std::fixed << std::setprecision(10) 
        << std::setw(colWidth)
        << json_data[argv[2]]["penyakit"][i].template get<std::string>()
        << " : " << output_data(i,0) << "\n";
    }    
    std::cout << "\n\n";


    // Creating a tabular data with pertrubation
    // using random normal distribution

    arma::mat training_data;
    const std::string data_path = std::string("data ")+argv[2]+".csv";
    mlpack::data::Load(data_path,training_data);

    arma::mat input_training_data = training_data.rows(0,input_col_size+1).t();
    arma::vec input_columns;        // vector of features
    std::vector<double> means;      // list means of features
    std::vector<double> stddevs;    // list standard deviations of features
    double mean_input_columns = 0;  // get current mean from feature 
    double stddev = 0;              // get current standard deviation from feature
    double Z = 0;                   // get sum square of (X_i - mean_input_columns) where the X_i is feature in index "i"
    
    for(uint32_t i = 0; i < input_col_size; i++){
        input_columns = input_training_data.col(i);
        mean_input_columns = arma::sum(input_columns) / input_columns.n_rows;
        
        Z = 0;
        for (auto& x : input_columns) {
            Z += pow(x - mean_input_columns, 2);
        }
        stddev = sqrt(Z/ input_columns.n_rows);
        means.push_back(mean_input_columns);
        stddevs.push_back(stddev);

    }

    arma::mat pertrubate_data;
    uint32_t total_pertrubate_data = 1;
    try {
        std::cout << "Enter total amount of pertrubate data: ";
        std::cin >> total_pertrubate_data;
        std::cin.ignore();
    }
    catch (std::exception& err) {
        std::cout << "Error : " << err.what() << "\n";
    }

    pertrubate_data.reshape(total_pertrubate_data, input_col_size);
    arma::mat transpose_input = input_data.t();
    for (uint32_t i = 0; i < means.size(); i++) {
        std::default_random_engine r_engine;
        std::normal_distribution<double> n_distribution(means[i], stddevs[i]);

        for (uint32_t j = 0; j < total_pertrubate_data; j++) {
            pertrubate_data(j, i) = transpose_input(0,i) + n_distribution(r_engine);
        }
    }

    arma::mat pertrubate_result;
    model.Predict(pertrubate_data.t(), pertrubate_result);
    pertrubate_result = pertrubate_result.t(); // size (total_pertrubate_data X output_row_size)

    // get the index class using argmax
    arma::mat selection_pertrubate_result = pertrubate_result;
    double pi_x = 0, distance = 0, sigma = 1, index_has_delete = 0;
    for (uint32_t i = 0; i < total_pertrubate_data; i++) {
        distance = arma::norm(output_data.t() - pertrubate_result.rows(i, i), 2);
        pi_x = exp(-pow(distance,2)/sigma);
        // check if pi_x distance are greater or equals to 0.5
        if (!(pi_x >= 0.7)) {
            pertrubate_data.shed_row(i - index_has_delete);
            selection_pertrubate_result.shed_row(i - index_has_delete);
            index_has_delete++;
        }
    }
    pertrubate_result.reset();

    // get single max output for each class from
    arma::mat vector_selection_pertrubate_result;
    arma::mat new_pertrubate_data, new_pertrubate_result;
    vector_selection_pertrubate_result.reshape(1, selection_pertrubate_result.n_rows);
    for (uint32_t i = 0; i < selection_pertrubate_result.n_rows; i++) {
        vector_selection_pertrubate_result(0, i) = selection_pertrubate_result.row(i).max();
    }
    selection_pertrubate_result.reset();

    // create g model, in this capstone project selection_pertrubate_result
    // i will use Logistic Regression

    pertrubate_data = pertrubate_data.t();
    mlpack::LinearRegression LR;
    LR.Train(pertrubate_data, vector_selection_pertrubate_result);
    std::cout << "\nParameters:\n";
    std::cout << LR.Parameters() << "\n\n";

    std::cout << "Pertrubate Data: \n";
    std::cout << pertrubate_data.t() << "\n\n";

    arma::mat validation;
    LR.Predict(pertrubate_data, validation);
    std::cout << "Validate : \n" << validation.t() << "\n";

    double INTERPRETABLE_VALUE = arma::norm(vector_selection_pertrubate_result - validation, 2);
    std::cout << "Interpretable value : " << pow(INTERPRETABLE_VALUE,2) << "\n";


    return EXIT_SUCCESS;
}

