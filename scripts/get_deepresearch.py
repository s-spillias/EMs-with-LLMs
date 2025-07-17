from gpt_researcher import GPTResearcher
import asyncio


async def get_deepresearch(query: str, report_type: str='deep', config_path=str='config.yaml') -> str:
    researcher = GPTResearcher(query, report_type)
    print('Starting deep research ...')
    research_data = await researcher.conduct_research()
    print('Research completed! Generating report ...')
    report = await researcher.write_report()
    return {
        'report': report,
        'source_urls': researcher.get_source_urls(),
        'research_costs': researcher.get_costs(),
    }


if __name__ == "__main__":
    query = 'Find X'
    report_type = 'deep'
    config_path = 'config_deepr.yaml'
    reports = asyncio.run(get_deepresearch(query, report_type, config_path=config_path))
    print("Report:\n", reports['report'])
    print(f"\nSource URLs: {reports['source_urls']}")
    print(f"\nResearch costs: {reports['research_costs']}")

