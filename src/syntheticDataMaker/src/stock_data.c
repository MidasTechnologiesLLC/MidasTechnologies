#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdint.h>
#include <time.h>
#include "stock_data.h"

// Recursively create directories
void mkdir_recursive(const char *path) {
    char temp[256];
    snprintf(temp, sizeof(temp), "%s", path);
    for (char *p = temp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(temp, 0755); // Create intermediate directories
            *p = '/';
        }
    }
    mkdir(temp, 0755); // Create the final directory
}

// Generate fake ticker IDs
char **generate_ticker_ids(const char *sector, int num_tickers) {
    char **tickers = malloc(num_tickers * sizeof(char *));
    if (!tickers) return NULL;

    for (int i = 0; i < num_tickers; i++) {
        tickers[i] = malloc(16); // Allocate space for a ticker ID
        if (!tickers[i]) return NULL;

        snprintf(tickers[i], 16, "%s%02d", sector, i + 1); // Create ticker ID
    }
    return tickers;
}

// Free allocated ticker IDs
void free_ticker_ids(char **tickers, int num_tickers) {
    for (int i = 0; i < num_tickers; i++) {
        free(tickers[i]);
    }
    free(tickers);
}

// Return sector description
const char *get_sector_description(const char *sector) {
    if (strcmp(sector, "Technology") == 0) return "technology and innovation";
    if (strcmp(sector, "Healthcare") == 0) return "medical and healthcare services";
    if (strcmp(sector, "Energy") == 0) return "energy production and distribution";
    if (strcmp(sector, "Finance") == 0) return "financial services and banking";
    if (strcmp(sector, "Consumer") == 0) return "consumer goods and retail";
    return "unknown sector";
}

// Generate synthetic stock data for 3 years at 5-minute intervals
int generate_stock_data(const char *ticker, const char *sector, double volatility, int days, int intervals_per_day) {
    char filename[256];
    snprintf(filename, sizeof(filename), "./data/%s/%s.json", sector, ticker);

    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Could not create file for ticker %s in sector %s.\n", ticker, sector);
        return 0;
    }

    fprintf(file, "[\n");

    double price = 100.0;  // Initial price
    long total_lines = 0;

    // Seed randomness
    srand((unsigned int)(time(NULL) ^ (uintptr_t)ticker));

    // Generate data for the specified number of days
    for (int day = 1; day <= days; day++) {
        for (int interval = 0; interval < intervals_per_day; interval++) { // 5-minute intervals
            double change = ((rand() % 200) - 100) / 100.0 * volatility; // Price change
            price += change;

            double open = price;
            double high = price + (rand() % 10) * volatility;
            double low = price - (rand() % 10) * volatility;
            double close = price + ((rand() % 20) - 10) * volatility;
            int volume = rand() % 10000 + 1000;

            // Generate time for the interval
            int hour = 9 + (interval * 5) / 60;
            int minute = (interval * 5) % 60;
            fprintf(file,
                    "  {\"date\": \"2024-%03d\", \"time\": \"%02d:%02d\", \"open\": %.2f, \"high\": %.2f, \"low\": %.2f, \"close\": %.2f, \"volume\": %d}%s\n",
                    day, hour, minute, open, high, low, close, volume,
                    (day == days && interval == intervals_per_day - 1) ? "" : ",");
            total_lines++;
        }
    }

    fprintf(file, "]\n");
    fclose(file);

    return total_lines; // Return the total number of lines written
}
