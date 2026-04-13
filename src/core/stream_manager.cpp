#include "core/stream_manager.hpp"
#include <algorithm>

namespace atos {
namespace core {

int StreamManager::addStream(const std::string& source) {
    std::lock_guard<std::mutex> lock(streams_mutex);
    int id = next_stream_id++;
    
    auto stream = std::make_unique<StreamData>();
    stream->id = id;
    stream->source = source;
    stream->active = true;
    
    streams[id] = std::move(stream);
    return id;
}

void StreamManager::removeStream(int streamId) {
    std::lock_guard<std::mutex> lock(streams_mutex);
    if (streams.count(streamId)) {
        streams[streamId]->active = false;
        streams.erase(streamId);
    }
}

void StreamManager::shutdown() {
    std::lock_guard<std::mutex> lock(streams_mutex);
    for (auto& pair : streams) {
        pair.second->active = false;
    }
    streams.clear();
}

std::vector<int> StreamManager::getActiveStreamIds() const {
    std::lock_guard<std::mutex> lock(streams_mutex);
    std::vector<int> ids;
    for (const auto& pair : streams) {
        ids.push_back(pair.first);
    }
    return ids;
}

} // namespace core
} // namespace atos
