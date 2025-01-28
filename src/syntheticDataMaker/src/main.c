#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include "stock_data.h"

#define NUM_SECTORS 8
#define NUM_TICKERS_PER_SECTOR 20
#define DAYS 1250
#define INTERVALS_PER_DAY 78 // 5-minute intervals in an 8-hour trading day

const char *sectors[] = {
    "Technology", "Healthcare", "Energy", "Finance",
    "Consumer", "Industrials", "Utilities", "RealEstate"
};

double sector_volatility[] = {2.0, 1.5, 2.5, 1.2, 1.0, 1.8, 0.8, 1.1};

// Shared metrics
typedef struct {
    int total_files_created;
    long total_lines_written;
    long total_volume_generated;
    pthread_mutex_t lock;
} Metrics;

Metrics metrics = {0, 0, 0, PTHREAD_MUTEX_INITIALIZER};

void *process_sector(void *args) {
    SectorArgs *sectorArgs = (SectorArgs *)args;

    char dir[256];
    snprintf(dir, sizeof(dir), "./data/%s", sectorArgs->sector);
    mkdir_recursive(dir);

    char readme_path[256];
    snprintf(readme_path, sizeof(readme_path), "%s/README.md", dir);
    FILE *readme_file = fopen(readme_path, "w");
    if (readme_file == NULL) {
        fprintf(stderr, "Error: Could not create README.md for sector %s.\n", sectorArgs->sector);
        pthread_exit(NULL);
    }
    fprintf(readme_file, "# %s Sector\n\n", sectorArgs->sector);
    fprintf(readme_file, "Volatility: %.2f\n", sectorArgs->volatility);
    fprintf(readme_file, "This sector represents companies involved in %s activities.\n", sectorArgs->sector_description);
    fclose(readme_file);

    for (int i = 0; i < sectorArgs->num_tickers; i++) {
        int lines_written = generate_stock_data(
            sectorArgs->tickers[i],
            sectorArgs->sector,
            sectorArgs->volatility,
            DAYS,
            INTERVALS_PER_DAY
        );

        // Update metrics
        pthread_mutex_lock(&metrics.lock);
        metrics.total_files_created++;
        metrics.total_lines_written += lines_written;
        pthread_mutex_unlock(&metrics.lock);
    }

    return NULL;
}

int main() {
    pthread_t threads[NUM_SECTORS];
    SectorArgs args[NUM_SECTORS];

    mkdir_recursive("./data");
    srand(time(NULL));

    for (int i = 0; i < NUM_SECTORS; i++) {
        args[i].sector = sectors[i];
        args[i].sector_description = get_sector_description(sectors[i]);
        args[i].num_tickers = NUM_TICKERS_PER_SECTOR;
        args[i].tickers = generate_ticker_ids(sectors[i], NUM_TICKERS_PER_SECTOR);
        args[i].volatility = sector_volatility[i];

        if (pthread_create(&threads[i], NULL, process_sector, &args[i]) != 0) {
            fprintf(stderr, "Error: Failed to create thread for sector %s.\n", sectors[i]);
            pthread_exit(NULL);
        }
    }

    for (int i = 0; i < NUM_SECTORS; i++) {
        pthread_join(threads[i], NULL);
        free_ticker_ids(args[i].tickers, NUM_TICKERS_PER_SECTOR);
    }

    printf("\n=== Metrics Summary ===\n");
    printf("Total files created: %d\n", metrics.total_files_created);
    printf("Total lines written: %ld\n", metrics.total_lines_written);
    printf("Total volume generated: %ld\n", metrics.total_volume_generated);
    printf("=======================\n");

    return 0;
}

