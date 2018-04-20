#include <glsp/huffman.hpp>
#include <list>
#include <algorithm>
#include <cstring>

namespace glshader::process::compress::huffman
{
    struct node {
        uint32_t f;
        uint8_t val;
        node* left;
        node* right;
        node* parent;
        int8_t tag;
        constexpr node(uint32_t f, uint8_t v, node* l, node* r, node* p, int8_t t)
            : f(f), val(v), left(l), right(r), parent(p), tag(t) {}
    };

    struct code {
        uint32_t length;
        std::list<uint8_t> signature;
    };

    std::vector<node> build_tree(const std::array<uint32_t, 256>& histogram, uint32_t non_zero_count)
    {
        struct node_comparator {
            constexpr bool operator()(node* one, node* other) const noexcept { return one->f > other->f; }
        };
        std::vector<node> nodes;
        nodes.reserve(2 * non_zero_count);
        std::priority_queue<node*, std::vector<node*>, node_comparator> queue;

        for (int i=0; i<256; ++i)
        {
            if (histogram[i] > 0)
            {
                nodes.emplace_back(histogram[i], uint8_t(i), nullptr, nullptr, nullptr, -1);
                queue.push(&nodes.back());
            }
        }

        while (!queue.empty())
        {
            node* left = queue.top();
            queue.pop();
            if (!queue.empty())
            {
                node* right = queue.top();
                queue.pop();

                nodes.emplace_back(left->f + right->f, 0, left, right, nullptr, -1);
                left->parent = &nodes.back();
                left->tag = 0;
                right->parent = &nodes.back();
                right->tag = 1;
                queue.push(&nodes.back());
            }
        }
        return nodes;
    }

    std::array<code, 256> generate_codes(const std::vector<node>& tree, const std::array<uint32_t, 256>& histogram, const uint32_t leaf_nodes)
    {
        std::array<code, 256> codes{ 0 };
        for (int i=0, ch = 0; i<256; ++i)
        {
            if (histogram[i]>0)
            {
                const node* n       = &tree[ch++];
                codes[i].signature.clear();
                codes[i].length     = 0;
                auto ins_it         = codes[i].signature.end();
                while (n->parent)
                {
                    ins_it = codes[i].signature.emplace(ins_it, n->tag & 0x1);
                    n = n->parent;
                    ++codes[i].length;
                }
            }
        }
        return codes;
    }
    stream encode(const std::basic_string<uint8_t>& in)
    {
        return encode(in.data(), in.size());
    }

    stream encode(const std::vector<uint8_t>& in)
    {
        return encode(in.data(), in.size());
    }

    stream encode(const uint8_t* in, size_t in_length)
    {
        uint32_t count = 0;
        std::array<uint32_t, 256> histogram{ 0 };
        std::basic_stringstream<uint8_t> stream;

        for (size_t i = 0; i < in_length; ++i)
        {
            if (histogram[in[i]] == 0)
                ++count;
            ++histogram[in[i]];
        }
        stream.write(reinterpret_cast<uint8_t*>(histogram.data()), sizeof(uint32_t) * histogram.size());

        std::vector<node>     nodes = build_tree(histogram, count);
        std::array<code, 256> codes = generate_codes(nodes, histogram, count);

        int bp = 0;
        uint8_t c = 0;
        const auto write_bit = [&stream, &bp, &c](uint8_t bit)
        {
            c = c | (bit << bp);
            bp = (bp+1) & 0x7;
            if (bp==0)
            {
                stream.put(c);
                c = 0;
            }
        };

        for (size_t x = 0; x < in_length; ++x)
            for (const auto bit : codes[in[x]].signature)
                write_bit(bit);

        if (c != 0)
            stream.put(c);

        stream.seekp(0, std::ios::end);
        size_t size = stream.tellp();
        stream.seekp(0, std::ios::beg);
        return { size, std::move(stream) };
    }

    stream decode(const std::basic_string<uint8_t>& in)
    {
        return decode(in.data(), in.size());
    }

    stream decode(const std::vector<uint8_t>& in)
    {
        return decode(in.data(), in.size());
    }

    stream decode(const uint8_t* in, size_t in_length)
    {
        if (in_length <= 256 * sizeof(uint32_t))
            return stream{ 0, std::basic_stringstream<uint8_t>{} };

        std::basic_stringstream<uint8_t> stream;
        std::array<uint32_t, 256> histogram{ 0 };
        uint32_t count = 0;
        std::memcpy(&histogram[0], in, 256*sizeof(uint32_t));
        int in_ptr = 256 * sizeof(uint32_t);
        for (int i=0; i<256; ++i)
        {
            count += uint32_t(histogram[i] != 0);
        }

        std::vector<node>     nodes = build_tree(histogram, count);
        std::array<code, 256> codes = generate_codes(nodes, histogram, count);

        int bp = 0;
        uint8_t c = in[in_ptr++];
        const auto read_bit = [&in, &in_length, &in_ptr, &bp, &c]()
        {
            uint8_t bit = (c >> bp) % 2;
            bp = (bp+1) & 0x7;
            if (bp==0)
            {
                if (in_ptr < in_length)
                    c = in[in_ptr++];
                else
                    c = 0;
            }
            return bit;
        };

        /* nodes.back() is the tree root. */
        uint32_t symb_count = nodes.back().f;
        while (symb_count > 0)
        {
            const node* n = &nodes.back();
            while (n->right)
            {
                const uint32_t bit = read_bit();
                n = bit ? n->right : n->left;
            }
            stream.put(n->val);
            --symb_count;
        }

        stream.seekp(0, std::ios::end);
        size_t size = stream.tellp();
        stream.seekp(0, std::ios::beg);
        return { size, std::move(stream) };
    }
}