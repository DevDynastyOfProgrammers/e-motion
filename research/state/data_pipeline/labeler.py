import asyncio
import os
import random
from typing import Any

import aiohttp
import pandas as pd
from loguru import logger

from research.fer_loader import config


class EmotionToGameParams:
    """Processor for converting emotion data to game parameters using AI API."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.api_url = config.API_URL
        self.headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

        self.presets: dict[str, list[float]] = {
            'god_mode': [-0.8, -0.9, -1.0, -1.0, 0.7, 1.0, 0.5],
            'walk_in_the_park': [-0.5, -0.5, -0.5, -0.5, 0.3, 0.5, 0.3],
            'beginner': [-0.2, -0.2, 0.0, -0.2, 0.0, 0.2, 0.1],
            'standard': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'challenge': [0.4, 0.2, 0.4, 0.3, 0.0, 0.0, 0.0],
            'survival_horror': [0.2, 0.0, 0.8, 0.8, -0.2, -0.3, -0.5],
            'nightmare': [0.8, 0.5, 0.7, 0.7, 0.3, 0.0, 0.4],
            'hardcore': [0.3, 0.6, 0.5, 1.0, 0.1, -0.5, -0.8],
            'bullet_heaven': [1.0, 0.8, 0.9, 0.9, 0.5, 0.8, 0.7],
            'impossible': [1.0, 1.0, 1.0, 1.0, -0.5, -1.0, -1.0],
        }

        self.game_parameters: list[str] = [
            'spawn_rate_multiplier',
            'enemy_speed_multiplier',
            'enemy_health_multiplier',
            'enemy_damage_multiplier',
            'player_speed_multiplier',
            'player_damage_multiplier',
            'item_drop_chance_modifier',
        ]

    def prepare_prompt(self, emotion_data: dict[str, float]) -> str:
        """Prepare AI prompt for emotion analysis."""
        emotion_info = '\n'.join([f'{k}: {v}' for k, v in emotion_data.items()])

        return f"""
You are an expert game psychologist specializing in dynamic difficulty adjustment. Analyze the player's emotional state and select the optimal game preset.

PLAYER EMOTIONAL STATE:
{emotion_info}

PRESET DIFFICULTY SPECTRUM (easiest to hardest):
1. god_mode (very easy) - for extreme frustration recovery
2. walk_in_the_park (easy) - for high stress relief  
3. beginner (slightly easy) - for moderate struggle support
4. standard (balanced) - for neutral emotional baseline
5. challenge (moderate) - for confident engagement
6. survival_horror (high survival) - for boredom prevention
7. nightmare (very hard) - for skilled challenge seekers
8. hardcore (extreme) - for masochistic players
9. bullet_heaven (fast-paced) - for excitement amplification
10. impossible (maximum) - for elite mastery testing

EXPERT DECISION MATRIX:

PRIMARY EMOTION DOMINANCE ANALYSIS:
- DOMINANT sad (>0.5): 
  * sad > 0.7 + confidence > 0.6 → god_mode (severe distress)
  * sad 0.5-0.7 → walk_in_the_park (moderate distress)
  * sad 0.4-0.5 → beginner (mild distress)

- DOMINANT angry_disgust (>0.4):
  * angry_disgust > 0.6 → beginner (high frustration needs relief)
  * angry_disgust 0.4-0.6 → challenge (moderate anger needs constructive outlet)

- DOMINANT fear_surprise (>0.4):
  * fear_surprise > 0.6 → walk_in_the_park (high anxiety needs safety)
  * fear_surprise 0.4-0.6 → beginner (moderate anxiety needs stability)

- DOMINANT happy (>0.4):
  * happy > 0.6 + confidence > 0.5 → survival_horror or nightmare (euphoric engagement)
  * happy 0.4-0.6 → challenge (positive engagement)

- DOMINANT neutral (>0.5):
  * neutral > 0.6 → standard (stable baseline)
  * neutral 0.5-0.6 → beginner or standard (slight engagement needed)

CONFIDENCE-BASED CALIBRATION:
- High confidence (>0.6): More decisive preset selection
- Medium confidence (0.4-0.6): Balanced approach
- Low confidence (<0.4): Prefer standard or beginner (uncertainty needs stability)

EMOTIONAL INTENSITY STRATIFICATION:
- CRITICAL NEGATIVE (any emotion >0.7): Maximum assistance needed
- HIGH NEGATIVE (0.5-0.7): Significant support required
- MODERATE NEGATIVE (0.3-0.5): Mild assistance beneficial
- MILD NEGATIVE (0.2-0.3): Slight adjustment

COMPOUND EMOTION PROTOCOLS:
- High sad (>0.5) + high fear_surprise (>0.4) → walk_in_the_park (emotional overload)
- High angry_disgust (>0.5) + low happy (<0.2) → beginner (frustration cycle)
- High happy (>0.5) + low negative (<0.3) → survival_horror (positive energy utilization)
- Balanced mixed emotions (all 0.2-0.4) → standard (complex equilibrium)
- High neutral (>0.6) + low variance (<0.2 range) → challenge (engagement stimulation)

PSYCHOLOGICAL MOMENTUM STRATEGY:
- Consecutive negative states: Gradual difficulty increase to prevent habituation
- Consecutive positive states: Maintain or slightly decrease challenge to avoid burnout
- Emotional volatility (high variance): Prefer standard for stability
- Emotional stability (low variance): Introduce variety with challenge/survival_horror

CONTEXT-AWARE DECISION HEURISTICS:
1. Prioritize emotional relief over challenge for severe negative states
2. Use confidence to weight decision certainty
3. Consider emotional complexity when multiple emotions are significant
4. Balance immediate needs with long-term engagement
5. Create controlled emotional arcs rather than random swings

RESPONSE FORMAT:
Return ONLY the preset name from the list above. No explanations.

Choose the optimal preset based on comprehensive emotional analysis:
"""

    async def call_grok_api(self, session: aiohttp.ClientSession, prompt: str) -> str | None:
        """Call Grok API with error handling."""
        payload = {
            'model': 'x-ai/grok-4.1-fast',
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.2,
            'max_tokens': 30,
        }

        try:
            async with session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.api_timeout),
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result['choices'][0]['message']['content'].strip()

        except asyncio.TimeoutError:
            logger.error('API call timed out')
            return None
        except aiohttp.ClientError as exc:
            logger.error(f'HTTP error during API call: {exc}')
            return None
        except Exception as exc:
            logger.error(f'Unexpected error during API call: {exc}')
            return None

    def parse_api_response(self, response: str) -> str | None:
        """Parse API response to extract preset name."""
        try:
            cleaned = response.strip().lower()
            cleaned = cleaned.replace('.', '').replace('"', '').replace("'", '').strip()

            for preset_name in self.presets.keys():
                if preset_name in cleaned:
                    return preset_name

            logger.warning(f'No valid preset found in response: {response}')
            return None

        except Exception as exc:
            logger.error(f'Error parsing response: {exc}, response: {response}')
            return None

    def add_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing required columns to DataFrame."""
        df_processed = df.copy()

        if 'preset' not in df_processed.columns:
            df_processed['preset'] = None

        for param in self.game_parameters:
            if param not in df_processed.columns:
                df_processed[param] = None

        return df_processed

    def apply_preset_values(self, df: pd.DataFrame, row_index: int, preset_name: str) -> None:
        """Apply preset values to DataFrame row."""
        if preset_name in self.presets:
            preset_values = self.presets[preset_name]
            for i, param_name in enumerate(self.game_parameters):
                df.at[row_index, param_name] = preset_values[i]

    def is_row_processed(self, row: pd.Series) -> bool:
        """Check if row already has preset and game parameters set."""
        # Check if preset is set and not None/NaN
        if pd.notna(row.get('preset')) and str(row.get('preset')).strip() != '':
            return True

        # Check if all game parameters are set and not None/NaN
        game_params_set = all(pd.notna(row.get(param)) and row.get(param) is not None for param in self.game_parameters)

        return game_params_set

    async def process_batch_concurrently(
        self, session: aiohttp.ClientSession, batch_data: list[dict[str, Any]], df: pd.DataFrame
    ) -> None:
        """Process a batch of rows concurrently."""
        tasks = []

        for row_data in batch_data:
            idx = row_data['index']
            emotion_data = row_data['emotion_data']
            prompt = self.prepare_prompt(emotion_data)
            task = self.call_grok_api(session, prompt)
            tasks.append((idx, task))

        # Execute all API calls concurrently
        results: list[tuple[int, str | None]] = []
        for idx, task in tasks:
            response = await task
            results.append((idx, response))

        # Process results
        successful_count = 0
        for idx, response in results:
            if response:
                preset_name = self.parse_api_response(response)
                if preset_name:
                    df.at[idx, 'preset'] = preset_name
                    self.apply_preset_values(df, idx, preset_name)
                    successful_count += 1
                    logger.debug(f"Row {idx}: Preset '{preset_name}' applied")
                else:
                    logger.warning(f'Row {idx}: Failed to parse preset from response')
            else:
                logger.error(f'Row {idx}: API call failed')

        logger.info(f'Batch completed: {successful_count}/{len(batch_data)} successful')

    async def process_emotions_batch_async(
        self,
        input_file: str,
        output_file: str | None = None,
        batch_size: int = 10,
        concurrent_requests: int = 5,
        delay_between_batches: float = 1.0,
        shuffle_batches: bool = True,
    ) -> None:
        """Process emotions batch asynchronously with progress tracking."""
        if output_file is None:
            output_file = input_file.replace('.csv', '_updated.csv')

        logger.info(f'📁 Loading data from {input_file}')
        df = pd.read_csv(input_file, delimiter=';')
        logger.info(f'✅ Loaded {len(df)} rows')

        df = self.add_missing_columns(df)
        logger.info('✅ Added missing columns')

        # Find unprocessed rows
        unprocessed_mask = ~df.apply(self.is_row_processed, axis=1)
        rows_to_process = df[unprocessed_mask]
        rows_count = len(rows_to_process)

        if rows_count == 0:
            logger.info('🎉 All rows already processed')
            return

        logger.info(f'🔧 Found {rows_count} unprocessed rows')

        # Enhanced shuffling with emotional diversity optimization
        if shuffle_batches and len(rows_to_process) > 0:
            emotional_columns = ['prob_angry_disgust', 'prob_fear_surprise', 'prob_happy', 'prob_neutral', 'prob_sad']
            rows_to_process = rows_to_process.copy()
            rows_to_process['emotional_complexity'] = rows_to_process[emotional_columns].std(axis=1)
            rows_to_process = rows_to_process.sample(frac=1, random_state=42, weights='emotional_complexity')
            logger.info('🔄 Rows shuffled with emotional complexity weighting')

        connector = aiohttp.TCPConnector(limit=concurrent_requests, limit_per_host=concurrent_requests)

        processed_count = 0
        total_rows = len(rows_to_process)

        async with aiohttp.ClientSession(connector=connector) as session:
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch = rows_to_process.iloc[start_idx:end_idx]
                batch_num = start_idx // batch_size + 1

                # Filter out already processed rows in this batch
                unprocessed_in_batch = []
                for idx, row in batch.iterrows():
                    if not self.is_row_processed(row):
                        unprocessed_in_batch.append((idx, row))

                if not unprocessed_in_batch:
                    logger.info(f'⏭️ Batch {batch_num}: All rows processed, skipping...')
                    continue

                logger.info(
                    f'🎯 Processing batch {batch_num} with {len(unprocessed_in_batch)} unprocessed rows '
                    f'({start_idx}-{end_idx - 1})'
                )

                # Prepare batch data only for unprocessed rows
                batch_data = []
                for idx, row in unprocessed_in_batch:
                    emotion_data = {
                        'true_class': row['true_class'],
                        'predicted_class': row['predicted_class'],
                        'confidence': row['confidence'],
                        'prob_angry_disgust': row['prob_angry_disgust'],
                        'prob_fear_surprise': row['prob_fear_surprise'],
                        'prob_happy': row['prob_happy'],
                        'prob_neutral': row['prob_neutral'],
                        'prob_sad': row['prob_sad'],
                    }
                    batch_data.append({'index': idx, 'emotion_data': emotion_data})

                # Process batch concurrently
                await self.process_batch_concurrently(session, batch_data, df)
                processed_count += len(batch_data)

                # Save progress after each batch
                df.to_csv(output_file, sep=';', index=False)
                logger.info(
                    f'💾 Progress saved to {output_file}. '
                    f'Processed {processed_count}/{total_rows} rows '
                    f'({processed_count / total_rows * 100:.1f}%)'
                )

                # Adaptive delay between batches
                if end_idx < total_rows and len(unprocessed_in_batch) > 0:
                    emotional_diversity = batch[emotional_columns].std().mean()
                    adaptive_delay = max(0.5, delay_between_batches * (1 - emotional_diversity))
                    random_delay = adaptive_delay + random.uniform(0, 0.5)

                    logger.info(f'⏳ Waiting {random_delay:.2f}s (diversity: {emotional_diversity:.3f})...')
                    await asyncio.sleep(random_delay)

        logger.info(f'🎉 Processing completed. Total processed: {processed_count} rows')

    def process_emotions_batch(
        self,
        input_file: str,
        output_file: str | None = None,
        batch_size: int = 10,
        concurrent_requests: int = 5,
        delay_between_batches: float = 1.0,
        shuffle_batches: bool = True,
    ) -> None:
        """Synchronous wrapper for async processing."""
        asyncio.run(
            self.process_emotions_batch_async(
                input_file, output_file, batch_size, concurrent_requests, delay_between_batches, shuffle_batches
            )
        )


def setup_logging() -> None:
    """Configure logging for the processor."""
    logger.remove()

    # File logging
    logger.add(
        'emotion_processor.log',
        level='INFO',
        format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
        rotation='10 MB',
        retention=5,
    )

    # Console logging
    logger.add(
        lambda msg: print(msg, end=''),
        level='INFO',
        format='<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>',
        colorize=True,
    )


def main() -> None:
    """Main execution function."""
    setup_logging()

    api_key = getattr(settings, 'OPENROUTER_API_KEY', '')
    if not api_key:
        logger.error('❌ OPENROUTER_API_KEY not found in settings')
        return

    processor = EmotionToGameParams(api_key)
    input_file = 'game_balancing_system/train/emotions.csv'
    output_file = 'game_balancing_system/train/emotions_updated.csv'

    if not os.path.exists(input_file):
        logger.error(f'❌ Input file not found: {input_file}')
        return

    logger.info('🚀 Starting emotion processing pipeline...')

    # Process with optimized parameters
    processor.process_emotions_batch(
        input_file=input_file,
        output_file=output_file,
        batch_size=15,
        concurrent_requests=5,
        delay_between_batches=1.0,
        shuffle_batches=True,
    )


if __name__ == '__main__':
    main()
