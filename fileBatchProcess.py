import asyncio
from concurrent.futures import ThreadPoolExecutor


async def main():
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, your_blocking_function)

asyncio.run(main())