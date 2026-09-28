#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace person {
template<class T> class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity):capacity_(capacity) {}
    bool push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock,[&]{return canceled_ || closed_ || queue_.size()<capacity_;});
        if(canceled_ || closed_) return false;
        queue_.push_back(std::move(value)); peak_=std::max(peak_,queue_.size());
        lock.unlock(); not_empty_.notify_one(); return true;
    }
    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock,[&]{return canceled_ || closed_ || !queue_.empty();});
        if(canceled_ || queue_.empty()) return false;
        value=std::move(queue_.front()); queue_.pop_front();
        lock.unlock(); not_full_.notify_one(); return true;
    }
    void close() { std::lock_guard<std::mutex> lock(mutex_); closed_=true; not_empty_.notify_all(); not_full_.notify_all(); }
    void cancel() { std::lock_guard<std::mutex> lock(mutex_); canceled_=true; not_empty_.notify_all(); not_full_.notify_all(); }
    size_t peak() const { std::lock_guard<std::mutex> lock(mutex_); return peak_; }
private:
    size_t capacity_,peak_{}; bool closed_{},canceled_{};
    std::deque<T> queue_; mutable std::mutex mutex_; std::condition_variable not_empty_,not_full_;
};
struct PipelineStats { size_t submitted{}, completed{}, inflight_peak{}, task_peak{}, result_peak{}, capacity{}; };
// A slot belongs to the producer, one worker, or the writer until it returns to free_slots.
// Completed results never block workers: the finite slot pool bounds the reorder map too.
template<class Task,class Result> class OrderedPipeline {
public:
    using Worker=std::function<Result(Task&)>;
    OrderedPipeline(std::vector<Worker> workers,size_t capacity):workers_(std::move(workers)),
        slots_(capacity),free_(capacity),tasks_(capacity) {
        if(workers_.empty() || !capacity) throw std::invalid_argument("Empty worker pool or capacity");
        stats_.capacity=capacity;
        for(size_t i=0;i<capacity;++i) free_.push(i);
    }
    void cancel(std::exception_ptr error) {
        { std::lock_guard<std::mutex> lock(mutex_); if(!error_) error_=error; canceled_=true; }
        free_.cancel(); tasks_.cancel(); changed_.notify_all();
    }
    void cancel(const std::string& reason) { cancel(std::make_exception_ptr(std::runtime_error(reason))); }
    template<class Read,class Write> PipelineStats run(Read read,Write write) {
        std::vector<std::thread> threads; std::thread writer;
        try {
            writer=std::thread([&]{
                try {
                    size_t next=0;
                    while(true) {
                        Entry entry;
                        {
                            std::unique_lock<std::mutex> lock(mutex_);
                            changed_.wait(lock,[&]{return canceled_ || results_.count(next) || done_==workers_.size();});
                            if(canceled_) break;
                            auto it=results_.find(next);
                            if(it==results_.end()) {
                                if(!results_.empty()) throw std::runtime_error("Missing frame in ordered pipeline");
                                break;
                            }
                            entry=std::move(it->second); results_.erase(it);
                        }
                        write(slots_[entry.slot],entry.result);
                        { std::lock_guard<std::mutex> lock(mutex_); ++stats_.completed; --inflight_; }
                        if(!free_.push(entry.slot)) break;
                        ++next;
                    }
                } catch(...) { cancel(std::current_exception()); }
            });
            for(size_t w=0;w<workers_.size();++w) threads.emplace_back([&,w]{
                try {
                    Work item;
                    while(tasks_.pop(item)) {
                        Result result=workers_[w](slots_[item.slot]);
                        std::lock_guard<std::mutex> lock(mutex_);
                        if(canceled_) break;
                        results_.emplace(item.ordinal,Entry{item.slot,std::move(result)});
                        stats_.result_peak=std::max(stats_.result_peak,results_.size()); changed_.notify_all();
                    }
                } catch(...) { cancel(std::current_exception()); }
                { std::lock_guard<std::mutex> lock(mutex_); ++done_; changed_.notify_all(); }
            });
            size_t slot,ordinal=0;
            while(free_.pop(slot)) {
                if(!read(slots_[slot],ordinal)) break;
                { std::lock_guard<std::mutex> lock(mutex_); ++stats_.submitted; ++inflight_;
                  stats_.inflight_peak=std::max(stats_.inflight_peak,inflight_); }
                if(!tasks_.push({ordinal,slot})) break;
                ++ordinal;
            }
        } catch(...) { cancel(std::current_exception()); }
        tasks_.close();
        for(auto& t:threads) t.join();
        if(writer.joinable()) writer.join();
        { std::lock_guard<std::mutex> lock(mutex_); if(error_) std::rethrow_exception(error_); }
        stats_.task_peak=tasks_.peak();
        if(stats_.submitted!=stats_.completed) throw std::runtime_error("Frame count mismatch");
        return stats_;
    }
private:
    struct Work { size_t ordinal{},slot{}; };
    struct Entry { size_t slot{}; Result result{}; };
    std::vector<Worker> workers_; std::vector<Task> slots_;
    BoundedQueue<size_t> free_; BoundedQueue<Work> tasks_;
    std::map<size_t,Entry> results_; std::mutex mutex_; std::condition_variable changed_;
    size_t done_{},inflight_{}; bool canceled_{}; std::exception_ptr error_; PipelineStats stats_;
};
}
