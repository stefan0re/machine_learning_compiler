#ifndef TENGEN_MINI_JIT_BACKEND_KERNEL_H
#define TENGEN_MINI_JIT_BACKEND_KERNEL_H

#include <sys/mman.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace TenGen::MiniJit::Backend {

    class Kernel {
       private:
        //! high-level code buffer
        std::vector<uint32_t> m_buffer;

        //! size of the kernel
        std::size_t m_size_alloc = 0;

        //! executable kernel
        void* m_kernel = nullptr;

        /**
         * Allocates memory through POSIX mmap.
         *
         * @param num_bytes number of bytes.
         **/
        void* alloc_mmap(std::size_t num_bytes) const {
            void* l_mem = mmap(0,
                               num_bytes,
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS,
                               -1,
                               0);

            if (l_mem == MAP_FAILED) {
                throw std::runtime_error("Failed to allocate memory: " + std::string(std::strerror(errno)));
            }

            return l_mem;
        }

        /**
         * Release POSIX mmap allocated memory.
         *
         * @param num_bytes number of bytes.
         * @param mem pointer to memory which is released.
         **/
        void release_mmap(std::size_t num_bytes,
                          void* mem) const {
            int l_res = munmap(mem,
                               num_bytes);

            if (l_res == -1) {
                throw std::runtime_error("Failed to release memory");
            }
        }

        /**
         * Sets the given memory region executable.
         *
         * @param num_bytes number of bytes.
         * @param mem point to memory.
         **/
        void set_exec(std::size_t num_bytes,
                      void* mem) const {
            int l_res = mprotect(mem,
                                 num_bytes,
                                 PROT_READ | PROT_EXEC);

            if (l_res == -1) {
                throw std::runtime_error("Failed to set memory executable: " + std::string(std::strerror(errno)));
            }
        }

        /**
         * Release memory of the kernel if allocated.
         **/
        void release_memory() {
            if (m_kernel != nullptr) {
                release_mmap(m_size_alloc,
                             m_kernel);
            }
            m_size_alloc = 0;

            m_kernel = nullptr;
        }

       public:
        /**
         * Constructor
         **/
        Kernel() {};

        /**
         * Destructor
         **/
        ~Kernel() noexcept {
            release_memory();
        }

        Kernel(Kernel const&) = delete;
        Kernel& operator=(Kernel const&) = delete;
        Kernel(Kernel&&) noexcept = delete;
        Kernel& operator=(Kernel&&) noexcept = delete;

        /**
         * Adds an instruction to the code buffer.
         *
         * @param ins instruction which is added.
         **/
        void add_instr(uint32_t ins) {
            m_buffer.push_back(ins);
        }

        /**
         * Gets the size of the code buffer.
         *
         * @return size of the code buffer in bytes.
         **/
        std::size_t get_size() const {
            return m_buffer.size() * 4;
        }

        /**
         * Sets the kernel based on the code buffer.
         **/
        void set_kernel() {
            release_memory();

            if (m_buffer.empty()) {
                return;
            }

            // alloc kernel memory
            m_size_alloc = m_buffer.size() * 4;
            try {
                m_kernel = (void*)alloc_mmap(m_size_alloc);
            } catch (std::runtime_error& e) {
                throw std::runtime_error("Failed to allocate memory for kernel: " + std::string(e.what()));
            }

            // copy instruction words from buffer to kernel memory
            for (std::size_t l_in = 0; l_in < m_buffer.size(); l_in++) {
                reinterpret_cast<uint32_t*>(m_kernel)[l_in] = m_buffer[l_in];
            }

            // clear cache
            char* l_kernel_ptr = reinterpret_cast<char*>(m_kernel);
            __builtin___clear_cache(l_kernel_ptr,
                                    l_kernel_ptr + m_buffer.size() * 4);

            // set executable
            set_exec(m_size_alloc,
                     m_kernel);
        }

        /**
         * Gets a pointer to the executable kernel.
         **/
        void const* get_kernel() const {
            return m_kernel;
        }

        /**
         * DEBUG: Writes the code buffer to the given file.
         *
         * @param path path to the file.
         * disassemble with: objdump -D -b binary -m aarch64 file.bin
         **/
        void write(char const* path) const {
            std::ofstream l_out(path,
                                std::ios::out | std::ios::binary);
            if (!l_out) {
                throw std::runtime_error("Failed to open file: " + std::string(path));
            }

            l_out.write(reinterpret_cast<char const*>(m_buffer.data()),
                        m_buffer.size() * 4);
        }

        /**
         * DEBUG: Clears the code buffer.
         **/
        void force_clear() {
            m_buffer.clear();
        }
    };

}  // namespace TenGen::MiniJit::Backend

#endif  // TENGEN_MINI_JIT_BACKEND_KERNEL_H