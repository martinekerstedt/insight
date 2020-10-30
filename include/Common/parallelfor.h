#ifndef PARALLELFOR_H
#define PARALLELFOR_H

#include <thread>
#include <algorithm>


template <typename func>
void parallel_for(func f, unsigned nb_elements)
{
#ifdef INSIGHT_DEBUG
    f(0, nb_elements);
#else

    // Get number of threads
    static unsigned nb_threads_hint = std::thread::hardware_concurrency();
    static unsigned nb_threads = nb_threads_hint == 0 ? 4 : nb_threads_hint;


    // Split evenly among threads
    unsigned batch_size = nb_elements / nb_threads;
    unsigned batch_remainder = nb_elements % nb_threads;

    std::vector<std::thread> my_threads(nb_threads);


    // Run threads
    for (unsigned i = 0; i < nb_threads; ++i) {

        int start = i * batch_size;

        my_threads[i] = std::thread(f, start, start+batch_size);
//        std::jthread(f, start, start+batch_size); // A lot slower
    }


    // Calc the remainders
    unsigned start = nb_threads * batch_size;
    f(start, start+batch_remainder);


    // Wait for all threads to finish
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));

#endif
}


#endif // PARALLELFOR_H
