#include "stdafx.h"
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>
#include <chrono>
#include <array>
#include <cstring>
#include <fstream>
#include <omp.h>
//Needed for usage of threading
#include <thread>

using namespace std;
using namespace std::chrono;

constexpr char *secret = "A long time ago in a galaxy far, far away....";

//Number of times to run a test
constexpr unsigned int TEST_ITERATIONS = 1;
constexpr unsigned int GENE_LENGTH = 8;
constexpr unsigned int NUM_COPIES_ELITE = 4;
constexpr unsigned int NUM_ELITE = 8;
constexpr double CROSSOVER_RATE = 0.9;
constexpr unsigned int POP_SIZE = 512;
const unsigned int NUM_CHARS = strlen(secret);
const unsigned int CHROMO_LENGTH = NUM_CHARS * GENE_LENGTH;
constexpr double MUTATION_RATE = 0.001;
//Create global variable to hold number of available threads for the system
int num_threads;

struct genome
{
	vector<unsigned int> bits;
	unsigned int fitness = 0.0;
	float gene_length = GENE_LENGTH;
};

genome best;

unsigned int calculate_total_fitness(const vector<genome> &genomes)
{
	unsigned int total_fitness = 0;

	//Holds temporary float of last value
	float result[4]{ 0,0,0,0 };

//Pre-Processor statement that initializes OpenMP's parallel for implementation
#pragma omp parallel for schedule(dynamic)
//SIMD version of calculate_total_fitness
	for (int i = 0; i < genomes.size(); i+=4)
	{
		//Generate temporary float to hold 4 values
		float temp[4]{ genomes[i].gene_length,genomes[i+1].gene_length,
			genomes[i+2].gene_length,genomes[i+3].gene_length};

		//Load float 
		__m128 value = _mm_loadu_ps(temp);
		//Load last value
		__m128 lastvalue = _mm_loadu_ps(&result[0]);
		//Add both values together 
		__m128 add = _mm_add_ps(lastvalue, value);
		//Store added value to local float
		_mm_storeu_ps(&result[0], add);
	}

	/*Original solution
	for (auto &genome : genomes)
		total_fitness += genome.gene_length;
	return total_fitness;
	*/

	//Sum up total fitness and return
	return result[0] + result[1] + result[2] + result[3];
}

inline bool comp_by_fitness(const genome &a, const genome &b)
{
	return a.fitness < b.fitness;
}

void grab_N_best(const unsigned int N, const unsigned int copies, vector<genome> &old_pop, vector<genome> &new_pop)
{
	sort(old_pop.begin(), old_pop.end(), comp_by_fitness);
	best = old_pop[0];
	for (unsigned int n = 0; n < N; ++n)
		for (unsigned int i = 0; i < copies; ++i)
			new_pop[n * copies + i] = old_pop[n];
}

const genome& roulette_wheel_selection(unsigned int pop_size, const unsigned int fitness, const vector<genome> &genomes)
{
	default_random_engine e(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
	static uniform_real_distribution<double> dist;
	double slice = dist(e) * fitness;
	unsigned int total = 0;

	//Holds temporary float of last value
	float result[4]{ 0,0,0,0 };

	//Pre-Processor statement that initializes OpenMP's parallel for implementation
//#pragma omp parallel for
	for (int i = 0; i < pop_size; i+=4)
	{
		//Temporary float to hold iterations of 4 
		float temp[4]{ genomes[i].fitness,genomes[i + 1].fitness,genomes[i + 2].fitness,
			genomes[i + 3].fitness };

		//Load float array
		__m128 value = _mm_loadu_ps(temp);
		//load last value
		__m128 lastvalue = _mm_loadu_ps(&result[0]);
		//carry out addition on two values
		__m128 add = _mm_add_ps(lastvalue, value);
		//Store the result in result float array
		_mm_storeu_ps(&result[0], add);
		//Set total to sum of result float array
		total = result[0] + result[1] + result[2] + result[3];

		if (total > slice)
			return genomes[i];
	}
	return genomes[0];
}

void cross_over(double crossover_rate, unsigned int chromo_length, const genome &mum, const genome &dad, genome &baby1, genome &baby2)
{
	default_random_engine e(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
	static uniform_real_distribution<double> float_dist;
	static uniform_int_distribution<unsigned int> int_dist(0, chromo_length);

	if (float_dist(e) > crossover_rate || mum.bits == dad.bits)
	{
		baby1.bits = mum.bits;
		baby2.bits = mum.bits;
	}
	else
	{
		const unsigned int cp = int_dist(e);

		baby1.bits.insert(baby1.bits.end(), mum.bits.begin(), mum.bits.begin() + cp);
		baby1.bits.insert(baby1.bits.end(), dad.bits.begin() + cp, dad.bits.end());
		baby2.bits.insert(baby2.bits.end(), dad.bits.begin(), dad.bits.begin() + cp);
		baby2.bits.insert(baby2.bits.end(), mum.bits.begin() + cp, mum.bits.end());
	}
}

void mutate(double mutation_rate, genome &gen)
{
    default_random_engine e(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
	static uniform_real_distribution<double> dist;
	double rnd;

	for (auto &bit : gen.bits)
	{
		rnd = dist(e);
		if (rnd < mutation_rate)
			bit = !bit;
	}
}

vector<genome> epoch(unsigned int pop_size, vector<genome> &genomes)
{
	auto fitness = calculate_total_fitness(genomes);
	vector<genome> babies(pop_size);

	if (((NUM_COPIES_ELITE * NUM_ELITE) % 2) == 0)
		grab_N_best(NUM_ELITE, NUM_COPIES_ELITE, genomes, babies);

//Pre-Processor statement that initializes OpenMP's parallel for implementation
#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
	for (int i = NUM_ELITE * NUM_COPIES_ELITE; i < pop_size; i += 2)
	{
		auto mum = roulette_wheel_selection(pop_size, fitness, genomes);
		auto dad = roulette_wheel_selection(pop_size, fitness, genomes);
		genome baby1;
		genome baby2;

		cross_over(CROSSOVER_RATE, CHROMO_LENGTH, mum, dad, baby1, baby2);
		mutate(MUTATION_RATE, baby1);
		mutate(MUTATION_RATE, baby2);
		babies[i] = baby1;
		babies[i + 1] = baby2;
	}
	return babies;
}

vector<unsigned int> decode(genome &gen)
{
	vector<unsigned int> decoded(NUM_CHARS);

	for (int gene = 0; gene < gen.bits.size(); gene += gen.gene_length)
	{
		//Calculate count
		int count = gene / gen.gene_length;

		//Local float to hold SIMD Value
		float val[4] = { 0,0,0,0 };
		float multiplier = 1;

		for (int c_bit = gen.gene_length; c_bit > 0; c_bit-=4)
		{
			//Calculate 4 iterations of multiplier
			float multi[4]{ multiplier,multiplier * 2,multiplier * 4,multiplier * 8 };
			//Get 4 iterations of gen and store in float array
			float temp[4]{ gen.bits[gene + c_bit],
				gen.bits[(gene + c_bit) - 1],
				gen.bits[(gene + c_bit) - 2],
				gen.bits[(gene + c_bit) - 3]};
			
			//Calculate addition then multiplication for val
			__m128 result = _mm_add_ps(_mm_loadu_ps(&val[0]) ,
				_mm_mul_ps(_mm_loadu_ps(&temp[0]), _mm_loadu_ps(&multi[0])));

			//Store result in float array
			_mm_storeu_ps(&val[0], result);

			//Update multiplier for next 4 iterations
			multiplier *= 16;
		}

		//Sum all of float array to get value for return
		decoded[count] = val[0] + val[1] + val[2] + val[3];
	}
	return decoded;



}

vector<vector<unsigned int>> update_epoch(unsigned int pop_size, vector<genome> &genomes)
{
	vector<vector<unsigned int>> guesses;
	genomes = epoch(pop_size, genomes);

	for (int i = 0; i < genomes.size(); ++i)
		guesses.push_back(decode(genomes[i]));
	return guesses;
}

unsigned int check_guess(const vector<unsigned int> &guess)
{
	vector<unsigned char> v(guess.size());

	for (unsigned int i = 0; i < guess.size(); ++i)
		v[i] = static_cast<unsigned char>(guess[i]);
	string s(v.begin(), v.end());
	unsigned int diff = 0;

	for (int i = 0; i < s.length(); ++i)
		diff += abs(s[i] - secret[i]);
	return diff;
}

string get_guess(const vector<unsigned int> &guess)
{
	vector<unsigned char> v(guess.size());
	int i; 

#pragma omp parallel for schedule(dynamic)
	for (i = 0; i < guess.size(); ++i)
		v[i] = static_cast<unsigned char>(guess[i]);
	string s(v.begin(), v.end());
	return s;
}

int main()
{
	//define number of available threads for parallisation
	num_threads = thread::hardware_concurrency();

	//Create file to store times in MS
	ofstream data("benchmark.csv", ofstream::out);

	//Being Testing with 100 iterations
	for (int i = 0; i < TEST_ITERATIONS; i++)
	{

		//Start Recording time
		auto start = system_clock::now();

		default_random_engine e(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
		uniform_int_distribution<unsigned int> int_dist(0, 1);
		vector<genome> genomes(POP_SIZE);

		for (int i = 0; i < POP_SIZE; ++i)
		{
			for (unsigned int j = 0; j < CHROMO_LENGTH; ++j)
				genomes[i].bits.push_back(int_dist(e));
		}
		auto population = update_epoch(POP_SIZE, genomes);

		for (int generation = 0; generation < 2048; ++generation)
		{
			for (int i = 0; i < POP_SIZE; ++i) 
				genomes[i].fitness = check_guess(population[i]);
			population = update_epoch(POP_SIZE, genomes);
			if (generation % 10 == 0)
			{
				cout << "Generation " << generation << ": " << get_guess(decode(best)) << endl;
				cout << "Diff: " << check_guess(decode(best)) << endl;
			}
		}

		//Print time CSV file in ms
		auto end = system_clock::now();
		auto total = end - start;
		cout << endl << duration_cast<milliseconds>(total).count() << endl;;
		data << duration_cast<milliseconds>(total).count() << endl;
	}

	//100 iterations complete. close file
	data.close();
	system("pause");
	return 0;
}