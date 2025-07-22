#include <string>
#include <vector>
#include <sstream> // Added for ostringstream
#include <fstream>
#include <iostream>
#include <unordered_map>


class OCRDictionary {
public:
    OCRDictionary() {}

    OCRDictionary(const std::string& dict_path) {
        this->load(dict_path);
    }
    OCRDictionary(const char* dict_path) {
        if (dict_path) { // Check for nullptr
            std::string dict_path_str(dict_path);
            this->load(dict_path_str);
        }
        else {
            std::cerr << "Error: Dictionary path is null." << std::endl;
        }
    }

    // 加载字典
    bool load(const std::string& dict_path) {
        std::ifstream file(dict_path);
        if (!file.is_open()) {
            std::cerr << "无法打开字典文件: " << dict_path << std::endl;
            return false;
        }
        std::string line;
        dictionary_.clear(); // Clear previous data if any
        char_to_index_.clear(); // Clear previous data if any
        while (std::getline(file, line)) {
            // Optional: Add UTF-8 BOM removal if necessary
            // if (line.rfind("\xEF\xBB\xBF", 0) == 0) { line.erase(0, 3); }
            if (!line.empty()) {
                dictionary_.push_back(line);
                char_to_index_[line] = static_cast<int>(dictionary_.size() - 1);
            }
        }
        file.close();
        // std::cout << "字典加载完成，字符数量: " << dictionary_.size() << std::endl;
        return true;
    }

    // 通过索引获取字符
    std::string get_char(int index) const {
        if (index < 0 || static_cast<size_t>(index) >= dictionary_.size()) {
            return " "; // 返回空字符串表示索引无效
        }
        return dictionary_[static_cast<size_t>(index)];
    }

    // 通过字符获取索引
    int get_index(const std::string& ch) const {
        auto it = char_to_index_.find(ch);
        if (it != char_to_index_.end()) {
            return it->second;
        }
        return -1; // -1 表示未找到
    }

    // 获取字典大小
    size_t size() const {
        return dictionary_.size();
    }

private:
    std::vector<std::string> dictionary_;          // 字符列表
    std::unordered_map<std::string, int> char_to_index_; // 反向查找表
};
